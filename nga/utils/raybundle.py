# from nerfstudio.engine.trainer import TrainerConfig
# from nerfstudio.configs.method_configs import method_configs
# from nerfstudio.configs.dataparser_configs import dataparsers as dataparser_configs
# from nerfstudio.utils.eval_utils import eval_load_checkpoint
from nerfstudio.cameras.rays import RayBundle
from nga.utils.spacial import convert_to_transformed_space

from PIL import Image
from pathlib import Path
import yaml
import json
import torch
from torch.nn.functional import normalize
import numpy as np
import OpenEXR, Imath


def plane_eval_ray_bundle(dataparser_outputs, sampling_depth, dimensions=(1.0,1.0), n = 1001):
    dataparser_scale = dataparser_outputs.dataparser_scale
    x = torch.linspace(-0.5*dimensions[0], 0.5*dimensions[0], n)
    y = torch.linspace(-0.5*dimensions[1], 0.5*dimensions[1], n)
    z = sampling_depth
    grid_x, grid_y = torch.meshgrid(x, y)
    origins = torch.stack([grid_x, grid_y, z * torch.ones([n, n])], dim=-1)
    origins = convert_to_transformed_space(origins, dataparser_outputs)
    directions = torch.zeros_like(origins)
    directions[:, :, 2] = -1.0
    pixel_area = (dataparser_scale ** 2) * torch.ones((n, n, 1)) / (n ** 2)
    nears = torch.zeros((n, n, 1))
    fars = torch.ones((n, n, 1)) * 2 * sampling_depth * dataparser_scale
    camera_indices = torch.zeros((n, n, 1))

    ray_bundle = RayBundle(
        origins=origins, directions=directions, pixel_area=pixel_area,
        camera_indices=camera_indices,
        nears=nears,
        fars=fars, 
    )
    return ray_bundle


def sphere_eval_ray_bundle(dataparser_outputs, sampling_depth, radius=0.5, n = 1001, m = 2001):
    dataparser_scale = dataparser_outputs.dataparser_scale
    phi = torch.linspace(0, np.pi, n)
    theta = torch.linspace(-np.pi, np.pi, m)
    
    grid_phi, grid_theta = torch.meshgrid(phi, theta)
    r = radius + sampling_depth
    x = r * torch.cos(grid_theta) * torch.sin(grid_phi)
    y = r * torch.sin(grid_theta) * torch.sin(grid_phi)
    z = r * torch.cos(grid_phi)
    origins = torch.stack([x, y, z], dim=-1)
    origins = convert_to_transformed_space(origins, dataparser_outputs)
    directions = -normalize(origins, dim=-1)
    pixel_area = (dataparser_scale ** 2) * torch.ones((n, m, 1)) / (n * n)
    nears = torch.zeros((n, m, 1))
    fars = torch.ones((n, m, 1)) * 2 * sampling_depth * dataparser_scale
    camera_indices = torch.zeros((n, m, 1))

    ray_bundle = RayBundle(
        origins=origins, directions=directions, pixel_area=pixel_area,
        camera_indices=camera_indices,
        nears=nears,
        fars=fars, 
    )
    return ray_bundle


def contour_eval_ray_bundle(dataparser_outputs, i, slice_count=10, dimensions=(1.0,1.0,1.0), n = 1001):
    dataparser_scale = dataparser_outputs.dataparser_scale
    x = torch.linspace(-0.5*dimensions[0], 0.5*dimensions[0], n)
    y = torch.linspace(-0.5*dimensions[1], 0.5*dimensions[1], n)
    sampling_depth = dimensions[2] / (slice_count)
    z = sampling_depth * ((slice_count)/2-i)
    grid_x, grid_y = torch.meshgrid(x, y)
    origins = torch.stack([grid_x, grid_y, z * torch.ones([n, n])], dim=-1)
    origins = convert_to_transformed_space(origins, dataparser_outputs)
    directions = torch.zeros_like(origins)
    directions[:, :, 2] = -1.0
    pixel_area = (dataparser_scale ** 2) * torch.ones((n, n, 1)) / (n ** 2)
    nears = torch.zeros((n, n, 1))
    fars = torch.ones((n, n, 1)) * sampling_depth * dataparser_scale
    camera_indices = torch.zeros((n, n, 1))

    ray_bundle = RayBundle(
        origins=origins, directions=directions, pixel_area=pixel_area,
        camera_indices=camera_indices,
        nears=nears,
        fars=fars, 
    )
    return ray_bundle


