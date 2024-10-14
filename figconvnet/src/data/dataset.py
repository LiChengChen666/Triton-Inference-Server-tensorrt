import logging
import concurrent.futures as cf
import functools
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from dgl.geometry import farthest_point_sampler
import random
from dataclasses import dataclass
from modulus.datapipes.datapipe import Datapipe
from torch.utils.data import Dataset
from modulus.datapipes.meta import DatapipeMetaData
from src.data.base_datamodule import BaseDataModule
try:
    import dgl
    from dgl.data import DGLDataset
except ImportError:
    raise ImportError(
        "Mesh Graph Net Datapipe requires the DGL library. Install the "
        + "desired CUDA version at: https://www.dgl.ai/pages/start.html"
    )
from torch.nn import functional as F
import scipy.sparse as sp

try:
    import pyvista as pv
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy
except ImportError:
    raise ImportError(
        "DrivAerNet Dataset requires the vtk and pyvista libraries. "
        "Install with pip install vtk pyvista"
    )

from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

logger = logging.getLogger(__name__)

def read_hdf_file(file_path):
    reader = vtk.vtkHDFReader()
    reader.SetFileName(file_path)
    reader.SetStep(0)
    reader.Update()
    return reader.GetOutput()

def save_json(var: Dict[str, torch.Tensor], file: str) -> None:
    """
    Saves a dictionary of tensors to a JSON file.

    Parameters
    ----------
    var : Dict[str, torch.Tensor]
        Dictionary where each value is a PyTorch tensor.
    file : str
        Path to the output JSON file.
    """
    var_list = {k: v.numpy().tolist() for k, v in var.items()}
    with open(file, "w") as f:
        json.dump(var_list, f)


def load_json(file: str) -> Dict[str, torch.Tensor]:
    """
    Loads a JSON file into a dictionary of PyTorch tensors.

    Parameters
    ----------
    file : str
        Path to the JSON file.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary where each value is a PyTorch tensor.
    """
    with open(file, "r") as f:
        var_list = json.load(f)
    var = {k: torch.tensor(v, dtype=torch.float) for k, v in var_list.items()}
    return var

@dataclass
class MetaData(DatapipeMetaData):
    name: str = "AhmedBody"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = False
    # Parallel
    ddp_sharding: bool = True


class CastingSteadyDataset(Dataset, Datapipe):
    """
    In-memory Ahmed body Dataset

    Parameters
    ----------
    data_dir: str
        The directory where the data is stored.
    split: str, optional
        The dataset split. Can be 'train', 'validation', or 'test', by default 'train'.
    num_samples: int, optional
        The number of samples to use, by default 10.
    invar_keys: Iterable[str], optional
        The input node features to consider. Default includes 'pos', '-Z_50mmpermin@Solid Transport@W', 'BC_Temperature_1220@Temperature@Temperature', 'p1', 'water_HTC4000_T40@Heat@Film Coeff'.
    outvar_keys: Iterable[str], optional
        The output features to consider. Default includes 'Effective Stress', 'Fraction Solid', 'Gap Width', 'Hot Tearing Indicator', 'Temperature' and 'Fluid Velocity'.
    normalize_keys Iterable[str], optional
        The features to normalize. Default includes all expect pos.
    normalization_bound: Tuple[float, float], optional
        The lower and upper bounds for normalization. Default is (-1, 1).
    force_reload: bool, optional
        If True, forces a reload of the data, by default False.
    name: str, optional
        The name of the dataset, by default 'dataset'.
    verbose: bool, optional
        If True, enables verbose mode, by default False.
    num_workers: int, optional
        Number of dataset pre-loading workers. If None, will be chosen automatically.
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_samples: int = 1,
        invar_keys: Iterable[str] = (
            'pos', 
            '-Z_50mmpermin@Solid Transport@W', 
            'BC_Temperature_1220@Temperature@Temperature', 
            'p1', 
            'water_HTC4000_T40@Heat@Film Coeff'
        ),
        outvar_keys: Iterable[str] = (
            'Effective Stress', 
            'Fraction Solid', 
            'Gap Width', 
            'Hot Tearing Indicator', 
            'Temperature',
            'Fluid Velocity'),
        normalize_keys: Iterable[str] = (
            'pos', 
            # '-Z_50mmpermin@Solid Transport@W', 
            # 'BC_Temperature_1220@Temperature@Temperature', 
            # 'p1', 
            # 'water_HTC4000_T40@Heat@Film Coeff',
            'Effective Stress', 
            'Fraction Solid', 
            'Gap Width', 
            'Hot Tearing Indicator', 
            'Temperature',
            'Fluid Velocity'
        ),
        normalization_bound: Tuple[float, float] = (-1.0, 1.0),
        force_reload: bool = False,
        name: str = "CastingSteadyDataset",
        verbose: bool = False,
        num_workers: Optional[int] = None,
    ):
        self.split = split
        self.num_samples = num_samples
        data_dir = Path(data_dir)
        self.data_dir = data_dir / self.split
        if not self.data_dir.is_dir():
            raise IOError(f"Directory not found {self.data_dir}")
        self.input_keys = list(invar_keys)
        self.output_keys = list(outvar_keys)
        self.normalize_keys = list(normalize_keys)
        self.normalization_bound = normalization_bound
        
        # Get case ids from the list of .vtp files.
        case_files = []
        for case_file in sorted(self.data_dir.glob("*.hdf")):
            case_files.append(str(case_file))

        self.length = min(len(case_files), self.num_samples)
        logging.info(f"Using {self.length} {split} samples.")

        if self.num_samples > self.length:
            raise ValueError(
                f"Number of available {self.split} dataset entries "
                f"({self.length}) is less than the number of samples "
                f"({self.num_samples})"
            )

        self.graphs = [None] * self.length
        if num_workers is None or num_workers <= 0:

            def get_num_workers():
                # Make sure we don't oversubscribe CPUs on a node.
                # TODO(akamenev): this should be in DistributedManager.
                local_node_size = max(
                    int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", 1)), 1
                )
                num_workers = len(os.sched_getaffinity(0)) // local_node_size
                return max(num_workers - 1, 1)

            num_workers = get_num_workers()
        with cf.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=torch.multiprocessing.get_context("spawn"),
        ) as executor:
            for (i, graph, coeff, normal, area) in executor.map(
                self.create_graph,
                range(self.length),
                case_files[: self.length],
                chunksize=max(1, self.length // num_workers),
            ):
                self.graphs[i] = graph
                
        # add the edge features
        # self.graphs = self.add_edge_features()

        # normalize the node and edge features
        if self.split == "train":
            self.node_stats = self._get_node_stats(keys=self.normalize_keys)
            # self.edge_stats = self._get_edge_stats()
        else:
            if not os.path.exists("node_stats.json"):
                raise FileNotFoundError(
                    "node_stats.json not found! Node stats must be computed on the training set."
                )
            # if not os.path.exists("edge_stats.json"):
            #     raise FileNotFoundError(
            #         "edge_stats.json not found! Edge stats must be computed on the training set."
            #     )
            self.node_stats = load_json("node_stats.json")
            # self.edge_stats = load_json("edge_stats.json")

        self.graphs = self.normalize_node()
        # self.graphs = self.normalize_edge()

    def create_graph(self, index: int, file_path: str) -> None:
        """Creates a graph from vtkhdf file.

        This method is used in parallel loading of graphs.

        Returns
        -------
            Tuple that contains graph index, graph, and optionally coeff, normal and area values.
        """
        vtudata = read_hdf_file(file_path)
        graph = self._create_dgl_graph(vtudata, self.output_keys, self.input_keys, dtype=torch.int32)

        coeff = None
        normal = None
        area = None
        return index, graph, coeff, normal, area

    @staticmethod
    def _create_dgl_graph(
        vtu: Any,
        outvar_keys: List[str],
        inputvar_keys: List[str],
        to_bidirected: bool = True,
        add_self_loop: bool = False,
        dtype: Union[torch.dtype, str] = torch.int32,
    ) -> dgl.DGLGraph:

        # Extract point data and connectivity information from the vtkPolyData
        points = vtu.GetPoints()
        if points is None:
            raise ValueError("Failed to get points from the polydata.")

        vertices = np.array(
            [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
        )

        # edge_list = []
        # for i in range(vtu.GetNumberOfCells()):
        #     cell = vtu.GetCell(i)
        #     for j in range(cell.GetNumberOfEdges()):
        #         edge = cell.GetEdge(j)
        #         p0 = edge.GetPointId(0)
        #         p1 = edge.GetPointId(1)
        #         edge_list.append((p0, p1))


        edge_list = []

        # 遍历所有单元格，处理四面体和六面体
        for i in range(vtu.GetNumberOfCells()):
            # cell_type = cell_types.GetValue(i)
            cell = vtu.GetCell(i)
            cell_type = cell.GetCellType()
            # print(cell.GetPointIds())
            cell_points = [cell.GetPointIds().GetId(j) for j in range(cell.GetNumberOfPoints())]

            # print(cell_type)
            # print(points)
            # 处理四面体 (cell_type == 10)
            if cell_type == vtk.VTK_TETRA:
                v0, v1, v2, v3 = cell_points
                edge_list.extend([
                    (v0, v1), (v0, v2), (v0, v3),
                    (v1, v2), (v1, v3), (v2, v3)
                ])
                

            # 处理六面体 (cell_type == 12)
            elif cell_type == vtk.VTK_WEDGE:
                v0, v1, v2, v3, v4, v5 = cell_points
                tetrahedra = [
                    (v0, v1, v2, v3),  
                    (v1, v2, v3, v5),  
                    (v1, v3, v4, v5),  
                ]
                for tetra in tetrahedra:
                    edge_list.extend([
                        (tetra[0], tetra[1]),
                        (tetra[1], tetra[2]),
                        (tetra[2], tetra[0]),
                        (tetra[0], tetra[3]),
                        (tetra[1], tetra[3]),
                        (tetra[2], tetra[3]),
                    ])
                # edge_list.extend([
                #     (v0, v1), (v1, v2), (v2, v0),  # 底面
                #     (v3, v4), (v4, v5), (v5, v3),  # 顶面
                #     (v0, v3), (v1, v4), (v2, v5),  # 侧面
                # ])

        # 移除重复的边
        edge_list = list(set(edge_list))
    
        # Create DGL graph using the connectivity information
        graph = dgl.graph(edge_list, idtype=dtype)
        if to_bidirected:
            graph = dgl.to_bidirected(graph)
        if add_self_loop:
            graph = dgl.add_self_loop(graph)

        # Assign node features using the vertex data
        obj_min = np.min(vertices, axis=0)
        obj_max = np.max(vertices, axis=0)
        obj_center = (obj_min + obj_max) / 2.0
        vertices = vertices - obj_center
        
        graph.ndata["pos"] = torch.tensor(vertices, dtype=torch.float32)
        # print(vertices)
        # Extract node attributes from the vtkPolyData
        point_data = vtu.GetPointData()
        if point_data is None:
            raise ValueError("Failed to get point data from the vtu.")

        for i in range(point_data.GetNumberOfArrays()):
            array = point_data.GetArray(i)
            array_name = array.GetName()
            if array_name in outvar_keys:
                array_data = np.zeros(
                    (points.GetNumberOfPoints(), array.GetNumberOfComponents())
                )
                for j in range(points.GetNumberOfPoints()):
                    array.GetTuple(j, array_data[j])
                # Assign node attributes to the DGL graph
                array_data = np.nan_to_num(array_data, nan=0)
                graph.ndata[array_name] = torch.tensor(array_data, dtype=torch.float32)
        
        field_data = vtu.GetFieldData()
        if field_data is None:
            raise ValueError("Failed to get field data from the vtu.")
        
        for i in range(field_data.GetNumberOfArrays()):
            array = field_data.GetArray(i)
            array_name = array.GetName()
            # print(array)
            if array_name in inputvar_keys:
                array_data = np.zeros(
                    (points.GetNumberOfPoints(), array.GetNumberOfComponents())
                )
                for j in range(points.GetNumberOfPoints()):
                    array.GetTuple(0, array_data[j])
                
                array_data = np.nan_to_num(array_data, nan=0)
                
                graph.ndata[array_name] = torch.tensor(array_data, dtype=torch.float32)
        return graph

    def add_edge_features(self) -> List[dgl.DGLGraph]:
        """
        Add relative displacement and displacement norm as edge features for each graph
        in the list of graphs. The calculations are done using the 'pos' attribute in the
        node data of each graph. The resulting edge features are stored in the 'x' attribute
        in the edge data of each graph.

        This method will modify the list of graphs in-place.

        Returns
        -------
        List[dgl.DGLGraph]
            The list of graphs with updated edge features.
        """
        if not hasattr(self, "graphs") or not self.graphs:
            raise ValueError("The list 'graphs' is empty.")

        for graph in self.graphs:
            pos = graph.ndata.get("pos")
            if pos is None:
                raise ValueError(
                    "'pos' does not exist in the node data of one or more graphs."
                )

            row, col = graph.edges()
            row = row.long()
            col = col.long()

            disp = pos[row] - pos[col]
            disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
            graph.edata["x"] = torch.cat((disp, disp_norm), dim=-1)

        return self.graphs

    def _get_edge_stats(self) -> Dict[str, Any]:
        """
        Computes the mean and standard deviation of each edge attribute 'x' in the
        graphs, and saves to a JSON file.

        Returns
        -------
        dict
            A dictionary with keys 'edge_mean' and 'edge_std' and the corresponding values being
            1-D tensors containing the mean or standard deviation value for each dimension of the edge attribute 'x'.
        """
        if not self.graphs:
            raise ValueError("The list 'graphs' is empty.")

        stats = {
            "edge_mean": 0,
            "edge_meansqr": 0,
        }
        for i in range(self.length):
            stats["edge_mean"] += (
                torch.mean(self.graphs[i].edata["x"], dim=0) / self.length
            )
            stats["edge_meansqr"] += (
                torch.mean(torch.square(self.graphs[i].edata["x"]), dim=0) / self.length
            )
        stats["edge_std"] = torch.sqrt(
            stats["edge_meansqr"] - torch.square(stats["edge_mean"])
        )
        stats.pop("edge_meansqr")

        # save to file
        save_json(stats, "edge_stats.json")
        return stats

    def _get_node_stats(self, keys: List[str]) -> Dict[str, Any]:
        """
        Computes the mean and standard deviation values of each node attribute
        for the list of keys in the graphs, and saves to a JSON file.

        Parameters
        ----------
        keys : list of str
            List of keys for the node attributes.

        Returns
        -------
        dict
            A dictionary with each key being a string of format '[key]_mean' or '[key]_std'
            and each value being a 1-D tensor containing the mean or standard deviation for each
            dimension of the node attribute.
        """
        if not self.graphs:
            raise ValueError("The list 'graphs' is empty.")

        stats = {}
        for key in keys:
            stats[key + "_mean"] = 0
            stats[key + "_meansqr"] = 0

        for i in range(self.length):
            for key in keys:
                stats[key + "_mean"] += (
                    torch.mean(self.graphs[i].ndata[key], dim=0) / self.length
                )
                stats[key + "_meansqr"] += (
                    torch.mean(torch.square(self.graphs[i].ndata[key]), dim=0)
                    / self.length
                )

        for key in keys:
            stats[key + "_std"] = torch.sqrt(
                stats[key + "_meansqr"] - torch.square(stats[key + "_mean"])
            ) + 0.000001
            stats.pop(key + "_meansqr")

        # save to file
        save_json(stats, "node_stats.json")
        return stats

    def normalize_node(self) -> List[dgl.DGLGraph]:
        """
        Normalize node data in each graph in the list of graphs.

        Returns
        -------
        List[dgl.DGLGraph]
            The list of graphs with normalized and concatenated node data.
        """
        if not hasattr(self, "graphs") or not self.graphs:
            raise ValueError("The list 'graphs' is empty.")

        if not hasattr(self, "node_stats") or not isinstance(self.node_stats, dict):
            raise ValueError(
                "The 'node_stats' attribute does not exist or is not a dictionary."
            )

        invar_keys = set(
            [
                key.replace("_mean", "").replace("_std", "")
                for key in self.node_stats.keys()
            ]
        )
        for i in range(len(self.graphs)):
            for key in invar_keys:
                self.graphs[i].ndata[key] = (
                    self.graphs[i].ndata[key] - self.node_stats[key + "_mean"]
                ) / self.node_stats[key + "_std"]

            self.graphs[i].ndata["x"] = torch.cat(
                [self.graphs[i].ndata[key] for key in self.input_keys], dim=-1
            )
            self.graphs[i].ndata["y"] = torch.cat(
                [self.graphs[i].ndata[key] for key in self.output_keys], dim=-1
            )
        return self.graphs

    def normalize_edge(self) -> List[dgl.DGLGraph]:
        """
        Normalize edge data 'x' in each graph in the list of graphs.

        Returns
        -------
        List[dgl.DGLGraph]
            The list of graphs with normalized edge data 'x'.
        """
        if not hasattr(self, "graphs") or not self.graphs:
            raise ValueError("The list 'graphs' is empty.")

        if not hasattr(self, "edge_stats") or not isinstance(self.edge_stats, dict):
            raise ValueError(
                "The 'edge_stats' attribute does not exist or is not a dictionary."
            )

        for i in range(len(self.graphs)):
            self.graphs[i].edata["x"] = (
                self.graphs[i].edata["x"] - self.edge_stats["edge_mean"]
            ) / self.edge_stats["edge_std"]
        return self.graphs

    def denormalize(self, pred, gt, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denormalize the graph node data.

        Parameters
        -----------
        pred: Tensor
            Normalized prediction
        gt: Tensor
            Normalized ground truth
        device: Any
            The device

        Returns
        --------
        Tuple(Tensor, Tensor)
            Denormalized prediction and ground truth
        """
        
        stats = self.node_stats
        stats = {key: val.to(device) for key, val in stats.items()}
        Stress = pred[..., [0]]
        Solid = pred[..., [1]]
        Gap = pred[..., [2]]
        Indicator = pred[..., [3]]
        Temperature = pred[..., [4]]
        Velocity = pred[..., 5:]

        Stress = Stress * stats["Effective Stress_std"] + stats["Effective Stress_mean"]
        Solid = Solid * stats["Fraction Solid_std"] + stats["Fraction Solid_mean"]
        Gap = Gap * stats["Gap Width_std"] + stats["Gap Width_mean"]
        Indicator = Indicator * stats["Hot Tearing Indicator_std"] + stats["Hot Tearing Indicator_mean"]
        Temperature = Temperature * stats["Temperature_std"] + stats["Temperature_mean"]
        Velocity = Velocity * stats["Fluid Velocity_std"] + stats["Fluid Velocity_mean"]
        
        Stress_gt = gt[..., [0]]
        Solid_gt = gt[..., [1]]
        Gap_gt = gt[..., [2]]
        Indicator_gt = gt[..., [3]]
        Temperature_gt = gt[..., [4]]
        Velocity_gt = gt[..., 5:]
        
        Stress_gt = Stress_gt * stats["Effective Stress_std"] + stats["Effective Stress_mean"]
        Solid_gt = Solid_gt * stats["Fraction Solid_std"] + stats["Fraction Solid_mean"]
        Gap_gt = Gap_gt * stats["Gap Width_std"] + stats["Gap Width_mean"]
        Indicator_gt = Indicator_gt * stats["Hot Tearing Indicator_std"] + stats["Hot Tearing Indicator_mean"]
        Temperature_gt = Temperature_gt * stats["Temperature_std"] + stats["Temperature_mean"]
        Velocity_gt = Velocity_gt * stats["Fluid Velocity_std"] + stats["Fluid Velocity_mean"]
        
        
        pred = torch.cat((Stress, Solid, Gap, Indicator, Temperature, Velocity), dim=-1)
        gt = torch.cat((Stress_gt, Solid_gt, Gap_gt, Indicator_gt, Temperature_gt, Velocity_gt), dim=-1)
        return pred, gt
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        return {'pos':graph.ndata['pos'],'x':graph.ndata['x'], 'y':graph.ndata['y']}

    def __len__(self):
        return self.length


class CastingSteadyDataModule(BaseDataModule):
    """DrivAerNet data module."""

    def __init__(
        self,
        data_path: str | Path,
        **kwargs,
    ):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        assert data_path.is_dir(), f"{data_path} is not a directory."

        self.data_path = data_path

        self._train_dataset = self._create_dataset("train")
        self._val_dataset = self._create_dataset("validation")
        self._test_dataset = self._create_dataset("test")

    def _create_dataset(self, prefix: str):
        # Create dataset with the processing pipeline.
        dataset = CastingSteadyDataset(self.data_path,prefix)
        
        return dataset
    
if __name__ == '__main__':
    dataset = CastingSteadyDataset('/workspace/project/bsms_steady/data')
    print(dataset.__len__())
    import time 
    t1 = time.time()
    dataset[0]
    t2 = time.time()
    print(t2-t1)
    
    datamodule = CastingSteadyDataModule('/workspace/project/bsms_steady/data')
    test_loader = datamodule.test_dataloader()
    print(test_loader)