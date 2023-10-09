"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

# from nga.template_datamanager import (
#     TemplateDataManagerConfig,
# )
# from nga.template_model import TemplateModelConfig

# from nga.models.depth_nerfacto import DepthNerfactoModelConfig
# from nga.models.generfacto import GenerfactoModelConfig
from nga.models.instant_ngp import InstantNGPModelConfig
from nga.models.mipnerf import MipNerfModel
from nga.models.nerfacto import NerfactoModelConfig
from nga.models.yuto import YutoModelConfig
from nga.models.jean import JeanModelConfig
# from nga.models.neus import NeuSModelConfig
# from nga.models.neus_facto import NeuSFactoModelConfig
# from nga.models.semantic_nerfw import SemanticNerfWModelConfig
from nga.models.tensorf import TensoRFModelConfig
from nga.models.vanilla_nerf import NeRFModel, VanillaModelConfig


from collections import OrderedDict
from typing import Dict

import tyro

from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.pixel_samplers import PairPixelSamplerConfig

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.configs.external_methods import get_external_methods

from nerfstudio.data.datamanagers.random_cameras_datamanager import RandomCamerasDataManagerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig

from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import InstantNGPDataParserConfig
from nga.data.dataparsers.nga_dataparser import NGADataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import PhototourismDataParserConfig
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.data.dataparsers.sitcoms3d_dataparser import Sitcoms3DDataParserConfig
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.data.datasets.sdf_dataset import SDFDataset
from nerfstudio.data.datasets.semantic_dataset import SemanticDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.sdf_field import SDFFieldConfig
# from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig
# from nerfstudio.models.generfacto import GenerfactoModelConfig
# from nerfstudio.models.instant_ngp import InstantNGPModelConfig
# from nerfstudio.models.mipnerf import MipNerfModel
# from nerfstudio.models.nerfacto import NerfactoModelConfig
# from nerfstudio.models.neus import NeuSModelConfig
# from nerfstudio.models.neus_facto import NeuSFactoModelConfig
# from nerfstudio.models.semantic_nerfw import SemanticNerfWModelConfig
# from nerfstudio.models.tensorf import TensoRFModelConfig
# from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nga.pipelines.nga_pipeline import NGAPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.plugins.registry import discover_methods
import math

method_configs: Dict[str, TrainerConfig] = {}
descriptions = {
    "nga-nerfacto": "Recommended real-time model tuned for real captures. This model will be continually updated.",
    "nga-yuto": "Recommended real-time model tuned for real captures. This model will be continually updated.",
    "nga-jean": "Hybrid of nerfacto and instant-ngp.",
    # "nga-depth-nerfacto": "Nerfacto with depth supervision.",
    "nga-instant-ngp": "Implementation of Instant-NGP. Recommended real-time model for unbounded scenes.",
    # "nga-instant-ngp-bounded": "Implementation of Instant-NGP. Recommended for bounded real and synthetic scenes",
    "nga-mipnerf": "High quality model for bounded scenes. (slow)",
    # "nga-semantic-nerfw": "Predicts semantic segmentations and filters out transient objects.",
    "nga-vanilla-nerf": "Original NeRF model. (slow)",
    "nga-tensorf": "tensorf",
    # "nga-dnerf": "Dynamic-NeRF model. (slow)",
    # "nga-phototourism": "Uses the Phototourism data.",
    # "nga-generfacto": "Generative Text to NeRF model",
    # "nga-neus": "Implementation of NeuS. (slow)",
    # "nga-neus-facto": "Implementation of NeuS-Facto. (slow)",
}
near_plane=0.5
far_plane=1+math.sqrt(3)

method_configs["nga-nerfacto"] = TrainerConfig(
    method_name="nga-nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=NGAPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NGADataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
            ),
        ),
        model=NerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            background_color="last_sample",
            near_plane=near_plane,
            far_plane=far_plane,
            disable_scene_contraction=True,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["nga-yuto"] = TrainerConfig(
    method_name="nga-yuto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=NGAPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NGADataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
            ),
        ),
        model=YutoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            background_color="last_sample",
            near_plane=near_plane,
            far_plane=far_plane,
            disable_scene_contraction=True,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["nga-jean"] = TrainerConfig(
    method_name="nga-jean",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=NGAPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NGADataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
            ),
        ),
        model=JeanModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            background_color="last_sample",
            near_plane=near_plane,
            far_plane=far_plane,
            disable_scene_contraction=True,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["nga-instant-ngp"] = TrainerConfig(
    method_name="nga-instant-ngp",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=NGAPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NGADataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
        ),
        model=InstantNGPModelConfig(
            eval_num_rays_per_chunk=8192,
            disable_scene_contraction=True,
            grid_levels=3,
            background_color="random",
            near_plane=near_plane,
            far_plane=far_plane,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
    vis="viewer",
)


method_configs["nga-mipnerf"] = TrainerConfig(
    method_name="nga-mipnerf",
    pipeline=NGAPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=NGADataParserConfig(), train_num_rays_per_batch=1024),
        model=VanillaModelConfig(
            _target=MipNerfModel,
            loss_coefficients={"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0},
            num_coarse_samples=128,
            num_importance_samples=128,
            eval_num_rays_per_chunk=1024,
            background_color="last_sample",
            near_plane=near_plane,
            far_plane=far_plane,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        }
    },
)

method_configs["nga-vanilla-nerf"] = TrainerConfig(
    method_name="nga-vanilla-nerf",
    pipeline=NGAPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NGADataParserConfig(),
        ),
        model=VanillaModelConfig(
            _target=NeRFModel,    
            background_color="last_sample",
            near_plane=near_plane,
            far_plane=far_plane,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
        "temporal_distortion": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
    },
)

method_configs["nga-tensorf"] = TrainerConfig(
    method_name="nga-tensorf",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=False,
    pipeline=NGAPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NGADataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
        ),
        model=TensoRFModelConfig(
            regularization="tv",
            background_color="last_sample",
            near_plane=near_plane,
            far_plane=far_plane,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=0.001),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
        },
        "encodings": {
            "optimizer": AdamOptimizerConfig(lr=0.02),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.002, max_steps=30000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

def make_method_spec(name):
    return MethodSpecification(
        config=method_configs[name],
        description=descriptions[name],
    )

nga_nerfacto = make_method_spec("nga-nerfacto")
nga_yuto = make_method_spec("nga-yuto")
nga_jean = make_method_spec("nga-jean")
nga_instant_ngp = make_method_spec("nga-instant-ngp")
nga_mipnerf = make_method_spec("nga-mipnerf")
nga_vanilla_nerf = make_method_spec("nga-vanilla-nerf")
nga_tensorf = make_method_spec("nga-tensorf")
