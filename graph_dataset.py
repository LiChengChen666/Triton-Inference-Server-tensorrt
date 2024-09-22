import logging
import functools
import json
import os
from pathlib import Path
import numpy as np
import torch
from dgl.geometry import farthest_point_sampler
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


class GraphDataset(DGLDataset):
    """In-memory MeshGraphNet Dataset for stationary mesh
    Parameters
    ----------
    name : str, optional
        Name of the dataset, by default "dataset"
    data_dir : _type_, optional
        Specifying the directory that stores the raw data in .TFRecord format., by default None
    split : str, optional
        Dataset split ["train", "eval", "test"], by default "train"
    invar_keys: Iterable[str], optional
        The input node features to consider. Default includes 'pos'.
    outvar_keys: Iterable[str], optional
        The output features to consider. Default includes 'p' and 'wallShearStress'.
    normalize_keys Iterable[str], optional
        The features to normalize. Default includes 'p' and 'wallShearStress'.
    cache_dir: str, optional
        Path to the cache directory to store graphs in DGL format for fast loading.
        Default is ./cache/.
    num_samples : int, optional
        Number of samples, by default 1000
    num_steps : int, optional
        Number of time steps in each sample, by default 600
    noise_std : float, optional
        The standard deviation of the noise added to the "train" split, by default 0.02
    force_reload : bool, optional
        force reload, by default False
    verbose : bool, optional
        verbose, by default False
    """
    def __init__(
        self,
        name="dataset",
        data_dir=None,
        split="train",
        point_data_keys = [],
        field_data_keys = [],
        normalize_keys = [],
        radius=1,
        input_window_size=1,
        output_window_size=1,
        noise_std=0.02,
        force_reload=False,
        verbose=False,
        sampling=False
    ):
        super().__init__(
            name=name,
            force_reload=force_reload,
            verbose=verbose,
        )
        self.data_dir = Path(data_dir)
        self.split = split
        self.noise_std = noise_std
        self.radius = radius
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.point_data_keys = point_data_keys
        self.field_data_keys = field_data_keys
        assert self.data_dir.is_dir(), f"错误，文件夹{self.data_dir}不存在"

        self.data_paths = []
        files = os.listdir(self.data_dir)
        for file in files:
            data_path = os.path.join(self.data_dir, file)
            self.data_paths.append(data_path)
            # logging.info(f'找到文件{data_path}')
        self.num_samples = len(self.data_paths)
        logging.info(f"轨迹个数：{self.num_samples}")

        self.readers = []
        self.steps = []
        for path in self.data_paths:
            reader = vtk.vtkHDFReader()
            reader.SetFileName(path)
            reader.UpdateInformation()
            self.steps.append(reader.GetNumberOfSteps())
            self.readers.append(reader)

        self.num_steps = min(self.steps)
        assert self.num_steps != 0, f"没有数据"

        self.length = self.num_samples * (self.num_steps - 1)

        self.samples = [np.arange(self.num_steps) for _ in range(self.num_samples)]

        init_pos = self.get_points([0],0)[0]
        self.sampling = sampling
        if(sampling == True):
            mesh_size =  np.random.randint(int(0.008*init_pos.shape[0]), int(0.01*init_pos.shape[0]))
            print('mesh_size:',mesh_size)
            while(mesh_size %10 !=0):
                    mesh_size += 1
            self.point_idx = farthest_point_sampler(init_pos.unsqueeze(0), mesh_size)[0]

            graph = dgl.knn_graph(init_pos[self.point_idx], 5)
            self.graph  = dgl.to_bidirected(graph)
        else:
            self.graph = self._create_dgl_graph(0)
        # print(self.graph)

    def __getitem__(self, idx):
        dp_idx = int(idx / self.num_steps)
        time_idx = int(idx % self.num_steps)
        in_idx = self.samples[dp_idx][time_idx]

        in_idxs = []
        for i in range(self.input_window_size):
            in_idx_temp = in_idx + i * 1
            if in_idx_temp > (self.num_steps - 1):
                in_idx_temp = self.num_steps - 1
            in_idxs.append(in_idx_temp)

        out_idxs = []
        for i in range(self.output_window_size):
            out_idx = in_idx + (i + 1) * self.input_window_size
            if out_idx > (self.num_steps - 1):
                out_idx = self.num_steps - 1
            out_idxs.append(out_idx)
        in_pos,out_pos,in_pos_data,out_pos_data,field_data= self.get_features(in_idxs,out_idxs,dp_idx)
        if self.sampling == True:
            in_pos = in_pos[:,self.point_idx]
            out_pos = out_pos[:,self.point_idx]
            in_pos_data = in_pos_data[:,self.point_idx]
            out_pos_data = out_pos_data[:,self.point_idx]

        field_data = field_data.flatten().unsqueeze(0).repeat(in_pos_data.size(1),1)

        dim = in_pos[-1].size(-1)
        # print(self.graph.edges()[0])
        edge_displacement = (torch.gather(in_pos[-1], dim=0, index=self.graph.edges()[0].unsqueeze(-1).expand(-1, dim)) -
                   torch.gather(in_pos[-1], dim=0, index=self.graph.edges()[1].unsqueeze(-1).expand(-1, dim)))
        if(self.radius is not None):
            edge_displacement /= self.radius
        edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)
        # print(edge_displacement)
        # print(in_pos_data.shape,in_pos.shape,field_data.shape)
        in_pos = in_pos.permute(1,0,2).reshape(in_pos_data.size(1), -1)

        node_features = torch.cat(  
            (in_pos_data.permute(1,0,2).reshape(in_pos_data.size(1), -1), in_pos, field_data), dim=-1
        )
        node_targets = torch.cat(
            (out_pos_data.permute(1,0,2).reshape(in_pos_data.size(1), -1), out_pos.permute(1,0,2).reshape(in_pos_data.size(1), -1)), dim=-1
        )
        edge_features = torch.cat((edge_displacement, edge_distance), dim=-1)
        # print(edge_features,edge_features.shape)
        
        self.graph.ndata["pos"] = in_pos
        self.graph.ndata["x"] = node_features
        self.graph.ndata["y"] = node_targets
        self.graph.edata['x'] = edge_features

        return self.graph
    
    def __len__(self):
        # 数据样本的数量
        return self.length

    def process(self):
        # 将原始数据处理为图、标签和数据集划分的掩码

        pass
    def save(self):
        # 将处理后的数据保存至 `self.save_path`
        pass

    def load(self):
        # 从 `self.save_path` 导入处理后的数据
        pass

    def get_features(self,input_idxs,output_idxs,dp_idx):
        idxs = input_idxs+output_idxs
        in_ends = len(input_idxs)
        pos = []
        pd_data = []
        fd_data = []
        for idx in idxs:
            reader = self.readers[dp_idx]
            reader.SetStep(idx)
            reader.Update()
            output = reader.GetOutput()
            field_data = output.GetFieldData()
            f = []
            for name in self.field_data_keys:
                data = vtk_to_numpy(field_data.GetArray(name))
                f.append(torch.tensor(data, dtype=torch.float))
            f = torch.cat(f, dim=-1)[None, ...]
            fd_data.append(f)
            points = torch.tensor(vtk_to_numpy(output.GetPoints().GetData())[None,...])
            pos.append(points)
            pd = []
            for key in self.point_data_keys:
                point_data = vtk_to_numpy(output.GetPointData().GetArray(key))
                if point_data.ndim == 1:
                    point_data = np.expand_dims(point_data, axis=-1)
                point_data = np.nan_to_num(point_data, nan=0)
                pd.append(torch.tensor(point_data))
            pd = torch.cat(pd, dim=-1)[None, ...]
            pd_data.append(pd)
        pos = torch.cat(pos, dim=0)
        pd_data = torch.cat(pd_data, dim=0)
        fd_data = torch.cat(fd_data, dim=0)
        # print(pos.shape,pd_data.shape,fd_data.shape)
        # print(input_idxs,output_idxs)
        in_pos = pos[:in_ends]
        out_pos = pos[in_ends:]
        in_pos_data = pd_data[:in_ends]
        out_pos_data = pd_data[in_ends:]
        in_field_data = fd_data[:in_ends]
        return in_pos,out_pos,in_pos_data,out_pos_data,in_field_data
    
    def get_points(self, idxs, dp_idx):
        xx = []
        tetra_cells = []
        hexa_cells = []
        for idx in idxs:
            reader = self.readers[dp_idx]
            reader.SetStep(idx)
            reader.Update()
            output = reader.GetOutput()
            points = vtk_to_numpy(output.GetPoints().GetData())[None,...]
            xx.append(torch.tensor(points))
            cells = output.GetCells()
            cell_types = output.GetCellTypesArray()

            for i in range(output.GetNumberOfCells()):
                cell_type = cell_types.GetValue(i)
                cell = output.GetCell(i)
                point_ids = cell.GetPointIds()

                if cell_type == vtk.VTK_TETRA:
                    tetra_cells.append([point_ids.GetId(j) for j in range(4)])  # 四面体4个顶点
                elif cell_type == vtk.VTK_HEXAHEDRON:
                    hexa_cells.append([point_ids.GetId(j) for j in range(8)])  # 六面体8个顶点
        tetra_cells = np.array(tetra_cells)
        hexa_cells = np.array(hexa_cells)
        def extract_edges_from_cells(cells, num_vertices):
            """从单元格提取边，num_vertices 表示每个单元的顶点数量"""
            edges = set()
            for cell in cells:
                for i in range(num_vertices):
                    for j in range(i + 1, num_vertices):
                        edges.add((cell[i], cell[j]))  # 无向边
                        edges.add((cell[j], cell[i]))  # 另一方向的无向边
            return edges
        tetra_edges = extract_edges_from_cells(tetra_cells, 4)  # 四面体有4个顶点
        hexa_edges = extract_edges_from_cells(hexa_cells, 8)   # 六面体有8个顶点

        # 合并四面体和六面体的边
        all_edges = np.array(list(tetra_edges | hexa_edges)).T
        
        xx = torch.cat(xx,dim=0)
        return xx[0], torch.tensor(all_edges)

    @staticmethod
    def add_edge_features(graph, pos):
        """
        adds relative displacement & displacement norm as edge features
        """
        row, col = graph.edges()
        disp = torch.tensor(pos[row.long()] - pos[col.long()])
        disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
        graph.edata["x"] = torch.cat((disp, disp_norm), dim=1)
        return graph

    @staticmethod
    def normalize_node(invar, mu, std):
        """normalizes a tensor"""
        if (invar.size()[-1] != mu.size()[-1]) or (invar.size()[-1] != std.size()[-1]):
            raise AssertionError("input and stats must have the same size")
        return (invar - mu.expand(invar.size())) / std.expand(invar.size())

    @staticmethod
    def normalize_edge(graph, mu, std):
        """normalizes a tensor"""
        if (
            graph.edata["x"].size()[-1] != mu.size()[-1]
            or graph.edata["x"].size()[-1] != std.size()[-1]
        ):
            raise AssertionError("Graph edge data must be same size as stats.")
        return (graph.edata["x"] - mu) / std

    @staticmethod
    def denormalize(invar, mu, std):
        """denormalizes a tensor"""
        denormalized_invar = invar * std + mu
        return denormalized_invar
    
    @staticmethod
    def _push_forward_diff(invar):
        return torch.tensor(invar[1:] - invar[0:-1], dtype=torch.float)

    @staticmethod
    def _add_noise(features, targets, noise_std, noise_mask):
        noise = torch.normal(mean=0, std=noise_std, size=features.size())
        noise_mask = noise_mask.expand(features.size()[0], -1, 2)
        noise = torch.where(noise_mask, noise, torch.zeros_like(noise))
        features += noise
        targets -= noise
        return features, targets
    
    def _create_dgl_graph(
        self,
        sample_id: int,
        to_bidirected: bool = True,
        dtype: torch.dtype | str = torch.int64,
    ) -> dgl.DGLGraph:
        
        reader = self.readers[sample_id]
        reader.SetStep(0)
        reader.Update()
        grid = reader.GetOutput()
        cells = grid.GetCells().GetData()
        points = vtk_to_numpy(grid.GetPoints().GetData())
        cell_indices = vtk_to_numpy(cells)
        edge = set()
        i = 0
        while i < len(cell_indices):
            num_points = cell_indices[i]
            indices = cell_indices[i+1:i+1+num_points]
            for j in range(num_points):
                for k in range(j+1,num_points):
                    edge.add((indices[j],indices[k]))
                    edge.add((indices[k],indices[j]))
            i += (num_points + 1)
        
        row, col = zip(*edge)
        data = np.ones(len(row))
        adj_matrix = sp.coo_matrix((data,(row,col)),shape=(points.shape[0],points.shape[0]))
        graph = dgl.from_scipy(adj_matrix,idtype=dtype) 
        if to_bidirected:
            graph = dgl.to_bidirected(graph)
        return graph
    
if __name__ == '__main__':
    import h5py as h5
    from trans import UnitGaussianNormalizer
    dataset = GraphDataset(data_dir='/home/h5',
                             point_data_keys=['Effective Stress','Temperature'],
                             field_data_keys=['Time','a'])
    with h5.File('/home/h5/out1.hdf', 'r') as f:
        Temperature = f['VTKHDF/PointData/Temperature'][:]
        Stress = f['VTKHDF/PointData/Effective Stress'][:]

    print(Temperature.shape)
        
    normalizers = UnitGaussianNormalizer.from_dataset(
                    [{'Temperature':torch.tensor(Temperature),'Stress':torch.tensor(Stress)}], dim=[1], keys=['Temperature','Stress']
                )

    print(normalizers['Temperature'].mean,normalizers['Temperature'].std)
    print(normalizers['Stress'].mean,normalizers['Stress'].std)

    import time 
    t1 = time.time()
    # a = dataset[10].edge_index - dataset[0].edge_index
    # is_all_zero = torch.sum(a)
    print(dataset[0].ndata['x'].shape)
    # for i in range(10):
    #     dataset[i]
    # print(is_all_zero)
    t2 = time.time()
    print((t2-t1)/10)