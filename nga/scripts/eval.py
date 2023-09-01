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
    config.print_to_terminal()
    return config

def next_eval(dataparser_scale, near_z, n = 1001):
    x = torch.linspace(-0.5, 0.5, n) * dataparser_scale
    y = torch.linspace(-0.5, 0.5, n) * dataparser_scale
    grid_x, grid_y = torch.meshgrid(x, y)
    origins = torch.stack([grid_x, grid_y, near_z * torch.ones([n, n])], dim=-1)
    directions = torch.zeros_like(origins)
    directions[:, :, 2] = -1.0
    pixel_area = torch.ones((n, n, 1)) / (n ** 2)
    camera_indices = torch.zeros((n, n, 1))
    ray_bundle = RayBundle(
        origins=origins, directions=directions, pixel_area=pixel_area, camera_indices=camera_indices
    )
    return ray_bundle, {}

def eval(config_path):
    config = load_config(config_path)

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.load_dir = config.get_checkpoint_dir()

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode="val")
    pipeline.eval()

    checkpoint_path, step = eval_load_checkpoint(config, pipeline)



    output_path = config.get_base_dir() / "results.json"
    render_output_path = config.get_base_dir() / "renders"

    render_output_path.mkdir(parents=True, exist_ok=True)

    path = config.get_base_dir() / "dataparser_transforms.json"

    data = json.load(open(path))

    dataparser_scale = data["scale"]
    near_z = dataparser_scale * 0.1
    transform = torch.tensor(data["transform"])


    camera_ray_bundle, batch = next_eval(dataparser_scale, near_z)
    camera_ray_bundle = camera_ray_bundle.to(device)
    height, width = camera_ray_bundle.shape
    num_rays = height * width
    outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

    rgb = outputs["rgb"]
    acc = colormaps.apply_colormap(outputs["accumulation"])
    depth = colormaps.apply_depth_colormap(
        torch.log(outputs["depth"]),
        accumulation=outputs["accumulation"],
    )
    Image.fromarray((rgb * 255).byte().cpu().numpy()).save(
        render_output_path / "rgb.jpg"
    )
    Image.fromarray((acc * 255).byte().cpu().numpy()).save(
        render_output_path / "acc.jpg"
    )
    Image.fromarray((depth * 255).byte().cpu().numpy()).save(
        render_output_path / "depth.jpg"
    )

    z = near_z - outputs["depth"]
    z /= dataparser_scale
    z -= transform[2,3]
    print(z)
    torch.save(z, render_output_path / "z.pt")

    # Get the output and define the names to save to
    benchmark_info = {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "checkpoint": str(checkpoint_path),
        "results": {
            "max_z": float(torch.max(z)),
            "min_z": float(torch.min(z)),
            "std_z": float(torch.std(z)),
            "mean_z": float(torch.mean(z)),
            "maz_abs_z": float(torch.max(torch.abs(z))),
        },
    }
    # Save output to output file
    output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")