from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.configs.dataparser_configs import dataparsers as dataparser_configs
from nerfstudio.utils.eval_utils import eval_load_checkpoint
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.utils import colormaps

from PIL import Image
from pathlib import Path
import yaml
import json
import torch

def load_config(config_path):
    with open(config_path, "r") as f:
        config_str = f.read()
    config = yaml.load(config_str, Loader=yaml.Loader)
    # config.print_to_terminal()
    return config

def convert_to_transformed_space(x, dataparser_transforms_data):
    dataparser_scale = dataparser_transforms_data["scale"]
    transform = torch.tensor(dataparser_transforms_data["transform"])
    x = torch.matmul(x, transform[:,0:3].transpose(0,1))
    x += transform[:,3]
    x *= dataparser_scale
    return x

def plane_eval_ray_bundle(dataparser_transforms_data, near_z, n = 1001):
    dataparser_scale = dataparser_transforms_data["scale"]
    x = torch.linspace(-0.5, 0.5, n)
    y = torch.linspace(-0.5, 0.5, n)
    z = near_z
    grid_x, grid_y = torch.meshgrid(x, y)
    origins = torch.stack([grid_x, grid_y, z * torch.ones([n, n])], dim=-1)
    origins = convert_to_transformed_space(origins, dataparser_transforms_data)
    directions = torch.zeros_like(origins)
    directions[:, :, 2] = -1.0
    pixel_area = (dataparser_scale ** 2) * torch.ones((n, n, 1)) / (n ** 2)
    nears = torch.zeros((n, n, 1))
    fars = torch.ones((n, n, 1)) * 2 * near_z * dataparser_scale
    camera_indices = torch.zeros((n, n, 1))

    ray_bundle = RayBundle(
        origins=origins, directions=directions, pixel_area=pixel_area,
        camera_indices=camera_indices,
        nears=nears,
        # fars=fars, 
    )
    return ray_bundle

def save_as_image(x, path):
    x = torch.clamp(x, 0, 1.0)
    Image.fromarray((x * 255).byte().cpu().numpy()).save(path)