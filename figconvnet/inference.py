import logging
from pathlib import Path
from src.models.figconvnet.geometries import GridFeaturesMemoryFormat
from dgl import DGLGraph
from dgl.dataloading import GraphDataLoader

from torch.utils.data import DataLoader, Dataset

import hydra
from hydra.utils import instantiate, to_absolute_path

import numpy as np
import pyvista as pv
import torch
import os
from omegaconf import DictConfig, OmegaConf

from modulus.distributed.manager import DistributedManager
from modulus.launch.utils import load_checkpoint

from loggers import init_python_logging
# from utils import batch_as_dict

from convert import get_cell,vtkHDFWriter

logger = logging.getLogger("agnet")


class EvalRollout:
    """MGN inference with a given experiment."""

    def __init__(self, cfg: DictConfig):
        self.output_dir = Path(to_absolute_path(cfg.output))
        logger.info(f"Storing results in {self.output_dir}")

        self.device = DistributedManager().device
        logger.info(f"Using {self.device} device")

        # instantiate dataset
        logger.info("Loading the test dataset...")
        # self.dataset = instantiate(cfg.data.test)
        # logger.info(f"Using {len(self.dataset)} test samples.")
        logger.info("Creating the model...")
        print(cfg.model)
        self.model = instantiate(cfg.model).to(self.device)
        
        self.datamodule = instantiate(cfg.data)
        self.dataloader = self.datamodule.test_dataloader(
            batch_size=cfg.eval.batch_size, **cfg.eval.dataloader
        )
    
        # instantiate the model

        # enable train mode
        self.model.eval()

        # load checkpoint
        chk = torch.load("/workspace/project/figconvnet/outputs/2024-10-13/15-26-51/model_00665.pth")
        self.model.load_state_dict(chk["model"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # instantiate losses.
        logger.info("Creating the losses...")
        self.loss = instantiate(cfg.loss)
        connectivity_array,cell_types_array,offset_array,num_cells,vertices = get_cell('/workspace/project/bsms_steady/data/validation/0.hdf')
        self.h = vtkHDFWriter('/workspace/project/figconvnet/outputs/t1.hdf',{'Effective Stress':1, 'Fraction Solid':1, 'Gap Width':1, 'Hot Tearing Indicator':1, 'Temperature':1, 'Fluid Velocity':3})
        self.h.init_geometry(connectivity_array,cell_types_array,offset_array,num_cells)

        
        self.points = vertices
        
    @torch.inference_mode()
    def predict(self, save_results=False):
        """
        Run the prediction process.

        Parameters:
        -----------
        save_results: bool
            Whether to save the results in form of a .vtp file, by default False

        Returns:
        --------
        None
        """

        for batch in self.dataloader:
            vertices = batch["pos"].float().to(self.device)  # (n_in, 3)
            featrues = batch['x'].float().to(self.device)
            print(featrues.shape)
            
            pred,drag_pred = self.model(vertices,featrues)
            gt = batch["y"].to(self.device)
            pred, gt = self.datamodule.test_dataset.denormalize(pred, gt, pred.device)
            d = {'Effective Stress':pred[0,:,0].cpu().numpy(), 'Fraction Solid':pred[0,:,1].cpu().numpy(), 'Gap Width':pred[0,:,2].cpu().numpy(), 
                    'Hot Tearing Indicator':pred[0,:,3].cpu().numpy(), 'Temperature':pred[0,:,4].cpu().numpy(), 'Fluid Velocity':pred[0,:,5:8].cpu().numpy()}
                #self.h.append(time,points,d)
                
                
            self.h.append(0,self.points,d)
            
            

def _init_python_logging(config: DictConfig) -> None:
    if config.log_dir is None:
        config.log_dir = config.output
    else:
        config.log_dir = to_absolute_path(config.log_dir)

    # Make the log dir
    os.makedirs(config.log_dir, exist_ok=True)

    # Set up Python loggers.
    if pylog_cfg := OmegaConf.select(config, "logging.python"):
        pylog_cfg.output = config.output
        pylog_cfg.rank = DistributedManager().rank
        # Enable logging only on rank 0, if requested.
        if pylog_cfg.rank0_only and pylog_cfg.rank != 0:
            pylog_cfg.handlers = {}
            pylog_cfg.loggers.figconv.handlers = []
        # Configure logging.
        logging.config.dictConfig(OmegaConf.to_container(pylog_cfg, resolve=True))


@hydra.main(version_base="1.3", config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    # initialize distributed manager
    DistributedManager.initialize()

    _init_python_logging(cfg)

    logger.info("Rollout started...")
    rollout = EvalRollout(cfg)
    rollout.predict(save_results=True)


def _init_hydra_resolvers():
    def res_mem_pair(
        fmt: str, dims: list[int, int, int]
    ) -> tuple[GridFeaturesMemoryFormat, tuple[int, int, int]]:
        return getattr(GridFeaturesMemoryFormat, fmt), tuple(dims)

    OmegaConf.register_new_resolver("res_mem_pair", res_mem_pair)
    
if __name__ == "__main__":
    _init_hydra_resolvers()
    
    main()
