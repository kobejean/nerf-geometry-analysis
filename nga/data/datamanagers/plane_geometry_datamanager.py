# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""
Data manager without input images, only random camera poses
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Union

import torch
from rich.progress import Console
from torch.nn import Parameter
from torch import Tensor
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataloaders import RandIndicesEvalDataloader

CONSOLE = Console(width=120)



@dataclass
class PlaneGeometryDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for data manager that does not load from a dataset. Instead, it generates random poses."""

    _target: Type = field(default_factory=lambda: PlaneGeometryDataManager)
    train_resolution: int = 64
    """Training resolution"""
    eval_resolution: int = 64
    """Evaluation resolution"""


class PlaneGeometryDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the VanillaDataManagerConfig used to instantiate class
    """

    config: PlaneGeometryDataManagerConfig

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        config: PlaneGeometryDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        VanillaDataManager.__init__(
            self,
            config,
            device,
            test_mode,
            world_size,
            local_rank,
            **kwargs,
        )



    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        n = self.config.eval_resolution
        x = torch.linspace(-0.5, 0.5, n)
        y = torch.linspace(-0.5, 0.5, n)
        grid_x, grid_y = torch.meshgrid(x, y)
        origins = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones(n * n)], dim=-1)
        directions = torch.zeros_like(origins)
        directions[:, 2] = -1.0
        pixel_area = torch.ones((n * n, 1)) / (n ** 2)
        camera_indices = torch.zeros((n * n, 1))
        ray_bundle = RayBundle(
            origins=origins, directions=directions, pixel_area=pixel_area, camera_indices=camera_indices
        )
        print(origins)
        return ray_bundle, {}
    
    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        n = self.config.eval_resolution
        x = torch.linspace(-0.5, 0.5, n)
        y = torch.linspace(-0.5, 0.5, n)
        grid_x, grid_y = torch.meshgrid(x, y)
        origins = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones(n * n)], dim=-1)
        directions = torch.zeros_like(origins)
        directions[:, 2] = -1.0
        pixel_area = torch.ones((n * n, 1)) / (n ** 2)
        camera_indices = torch.zeros((n * n, 1))
        ray_bundle = RayBundle(
            origins=origins, directions=directions, pixel_area=pixel_area, camera_indices=camera_indices
        )
        print(origins)

        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, ray_bundle, batch
        raise ValueError("No more eval images")

