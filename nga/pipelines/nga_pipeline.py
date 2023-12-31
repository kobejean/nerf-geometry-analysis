"""
Nerfstudio Template Pipeline
"""

from matplotlib.animation import FuncAnimation
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import typing
import torch
import glob
from dataclasses import dataclass, field
from typing import Literal, Optional, Type
from pathlib import Path
import subprocess

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nga.utils.utils import (
    load_config,
    save_as_image, read_depth_map, compute_errors,
    load_metadata
)
from nga.utils.raybundle import contour_eval_ray_bundle, plane_eval_ray_bundle, sphere_eval_ray_bundle, cube_eval_ray_bundle
from nga.utils.spacial import convert_from_transformed_space
# from nga.template_datamanager import TemplateDataManagerConfig
# from nga.template_model import TemplateModel, TemplateModelConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nga.models.base_model import ModelConfig
from nga.utils.eval_vis import (
    save_contour_renders,
    save_depth_vis,
    save_weight_distribution_plot,
    save_geometry_surface_eval,
    eval_set_renders_and_metrics,
)
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import colormaps


@dataclass
class NGAPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: NGAPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = DataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ModelConfig()
    """specifies the model config"""

class NGAPipeline(VanillaPipeline):
    """Template Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """ 

    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        # metrics_dict = super().get_average_eval_image_metrics(step, output_path, get_std)
        metrics_dict = {}

        self.eval()

        depth_filenames = self.datamanager.eval_dataset.metadata["depth_filenames"]
        method_name = self.model.__class__.__name__
        geometry_analysis_type = self.datamanager.train_dataparser_outputs.metadata.get("geometry_analysis_type", "unspecified")
        geometry_analysis_dimensions = self.datamanager.train_dataparser_outputs.metadata.get("geometry_analysis_dimensions", {}) 


        if geometry_analysis_type != "unspecified":
            save_geometry_surface_eval(self, output_path, padded=True)
            
            surface_metrics = save_geometry_surface_eval(self, output_path)
            metrics_dict = { **metrics_dict, **surface_metrics }

        # if output_path is not None:
        #     save_contour_renders(self, output_path, slice_count=20)
            
        metrics_dict = { **metrics_dict , **eval_set_renders_and_metrics(self, output_path, get_std) }
        

        self.train()
        return metrics_dict