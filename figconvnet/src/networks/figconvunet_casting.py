# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib
import torch
from torch import Tensor

matplotlib.use("Agg")  # use non-interactive backend
import matplotlib.pyplot as plt

from src.models.figconvnet.figconvunet import FIGConvUNet

from src.models.figconvnet.geometries import (
    GridFeaturesMemoryFormat,
)

from src.models.figconvnet.components.reductions import REDUCTION_TYPES

from src.utils.visualization import fig_to_numpy
from src.utils.eval_funcs import eval_all_metrics


class FIGConvUNetCastingNet(FIGConvUNet):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        num_down_blocks: Union[int, List[int]] = 1,
        num_up_blocks: Union[int, List[int]] = 1,
        mlp_channels: List[int] = [512, 512],
        aabb_max: Tuple[float, float, float] = (2.5, 1.5, 1.0),
        aabb_min: Tuple[float, float, float] = (-2.5, -1.5, -1.0),
        voxel_size: Optional[float] = None,
        resolution_memory_format_pairs: List[
            Tuple[GridFeaturesMemoryFormat, Tuple[int, int, int]]
        ] = [
            (GridFeaturesMemoryFormat.b_xc_y_z, (2, 128, 128)),
            (GridFeaturesMemoryFormat.b_yc_x_z, (128, 2, 128)),
            (GridFeaturesMemoryFormat.b_zc_x_y, (128, 128, 2)),
        ],
        use_rel_pos: bool = True,
        use_rel_pos_encode: bool = True,
        pos_encode_dim: int = 32,
        communication_types: List[Literal["mul", "sum"]] = ["sum"],
        to_point_sample_method: Literal["graphconv", "interp"] = "graphconv",
        neighbor_search_type: Literal["knn", "radius"] = "knn",
        knn_k: int = 16,
        reductions: List[REDUCTION_TYPES] = ["mean"],
        drag_loss_weight: Optional[float] = None,
        pooling_type: Literal["attention", "max", "mean"] = "max",
        pooling_layers: List[int] = None,
    ):
        super().__init__(
            in_channels=hidden_channels[0],
            out_channels=out_channels,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            num_down_blocks=num_down_blocks,
            num_up_blocks=num_up_blocks,
            mlp_channels=mlp_channels,
            aabb_max=aabb_max,
            aabb_min=aabb_min,
            voxel_size=voxel_size,
            resolution_memory_format_pairs=resolution_memory_format_pairs,
            use_rel_pos=use_rel_pos,
            use_rel_pos_embed=use_rel_pos_encode,
            pos_encode_dim=pos_encode_dim,
            communication_types=communication_types,
            to_point_sample_method=to_point_sample_method,
            neighbor_search_type=neighbor_search_type,
            knn_k=knn_k,
            reductions=reductions,
            drag_loss_weight=drag_loss_weight,
            pooling_type=pooling_type,
            pooling_layers=pooling_layers,
        )

    def data_dict_to_input(self, data_dict) -> torch.Tensor:

        vertices = data_dict["pos"].float()  # (n_in, 3)
        featrues = data_dict['x'].float()

       
        # Assume it is centered
        # center vertices
        # vertices_max = vertices.max(1)[0]
        # vertices_min = vertices.min(1)[0]
        # vertices_center = (vertices_max + vertices_min) / 2.0
        # vertices = vertices - vertices_center

        return vertices.to(self.device),featrues.to(self.device)

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, datamodule=None, **kwargs) -> Dict:
        vertices,featrues = self.data_dict_to_input(data_dict)
        normalized_pred, drag_pred = self(vertices,featrues)
        if loss_fn is None:
            loss_fn = self.loss
        normalized_gt = (
            data_dict["y"]
            .to(self.device)
            .view_as(normalized_pred)
        )
        out_dict = {"l2": loss_fn(normalized_pred, normalized_gt)}

        pred, gt = datamodule.train_dataset.denormalize(normalized_pred.clone(),normalized_gt.clone(),self.device)
        # gt = data_dict["y"].to(self.device).view_as(pred)
        out_dict["l2_decoded"] = loss_fn(pred, gt)
        # Pressure evaluation
        out_dict.update(
            eval_all_metrics(normalized_gt, normalized_pred, prefix="norm_pressure")
        )
        # collect all drag outputs. All _ prefixed keys are collected in the meter
        # gt_drag = data_dict["c_d"].float()
        # out_dict["_gt_drag"] = gt_drag.cpu().flatten()
        # out_dict["_pred_drag"] = drag_pred.detach().cpu().flatten()
        return out_dict

    def loss_dict(self, data_dict, loss_fn=None, datamodule=None, **kwargs) -> Dict:
        vertices,featrues = self.data_dict_to_input(data_dict)

        normalized_pred, drag_pred = self(vertices,featrues)
        
        normalized_gt = data_dict["y"].to(self.device)

        return_dict = {}
        if loss_fn is None:
            loss_fn = self.loss
        # print(loss_fn)
        return_dict["l2loss"] = loss_fn(
            normalized_pred.view(1, -1), normalized_gt.view(1, -1).to(self.device)
        )

        # compute drag loss
        drag_loss_fn = loss_fn
        # if drag_loss_fn is in self attribute
        if hasattr(self, "drag_loss_fn"):
            drag_loss_fn = self.drag_loss_fn

        # gt_drag = data_dict["c_d"].float().to(self.device)
        # return_dict["drag_loss"] = drag_loss_fn(drag_pred, gt_drag.view_as(drag_pred))

        # if drag weight is in self attribute
        if hasattr(self, "drag_weight"):
            return_dict["drag_loss"] *= self.drag_weight

        return return_dict





def drivaer_create_subplot(ax, vertices, data, title):
    # Flip along x axis
    vertices = vertices.clone()
    vertices[:, 0] = -vertices[:, 0]

    sc = ax.scatter(
        vertices[:, 0], vertices[:, 1], vertices[:, 2], c=data, cmap="viridis"
    )
    # Make the colorbar smaller
    # fig.colorbar(sc, ax=ax, shrink=0.25, aspect=5)
    # Show the numbers on the colorbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.25, aspect=5)
    cbar.set_label(title, rotation=270, labelpad=20)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal")
    # remove grid and background
    ax.grid(False)
    # ax.xaxis.pane.set_edgecolor('black')
    # ax.yaxis.pane.set_edgecolor('black')
    # remove bounding wireframe
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # remove all ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
