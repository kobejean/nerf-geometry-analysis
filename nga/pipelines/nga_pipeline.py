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
from dataclasses import dataclass, field
from typing import Literal, Optional, Type
from pathlib import Path

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nga.utils.utils import (
    load_config, plane_eval_ray_bundle, sphere_eval_ray_bundle, 
    save_as_image, read_depth_map, compute_errors,
    load_metadata
)
from nga.utils.spacial import convert_from_transformed_space
# from nga.template_datamanager import TemplateDataManagerConfig
# from nga.template_model import TemplateModel, TemplateModelConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nga.models.base_model import ModelConfig
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

def save_weight_distribution_plot(filepath, outputs, xlim = [0, 6], dataparser_scale = 1.0, plot_cdf = False):
    if "weight_hist" in outputs:
        assert "weight_hist_edges" in outputs

        # Create figure object
        fig, ax = plt.subplots()

        # Create bar chart
        dist = torch.cumsum(outputs["weight_hist"], 0) if plot_cdf else outputs["weight_hist"]
        edges = outputs["weight_hist_edges"] / dataparser_scale
        ax.stairs(dist, edges, fill=True)

        # Set axis labels and title
        ax.set_xlabel('Depth')
        ax.set_ylabel('Weight')
        ax.set_title('Weight CDF' if plot_cdf else 'Weight Histogram')
        # ax.set_xlim(xlim)


        # Save the figure
        fig.savefig(filepath)
        plt.close(fig)

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


        def save_depth_vis(file_path, depth_gt, depth_pred, depth_diff, mask):
            # Convert torch tensors to numpy arrays and apply mask
            depth_gt_np = depth_gt.to("cpu").numpy()
            depth_pred_np = depth_pred.to("cpu").numpy()
            depth_diff_np = depth_diff.to("cpu").numpy()

            mask_np = mask.to("cpu").numpy()

            depth_gt_masked = np.ma.masked_where(mask_np == 0, depth_gt_np)
            depth_pred_masked = depth_pred_np
            depth_diff_masked = np.ma.masked_where(mask_np == 0, depth_diff_np)
            # Find min and max depth values for a unified color scale
            vmin = min(np.min(depth_gt_masked), np.min(depth_pred_masked))
            vmax = max(np.max(depth_gt_masked), np.max(depth_pred_masked))

            # Create the subplots
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(method_name)

            # Plot the heatmaps
            cax1 = axs[0].imshow(depth_gt_masked, cmap='viridis', vmin=vmin, vmax=vmax)
            axs[0].set_title('depth_gt')
            fig.colorbar(cax1, ax=axs[0])

            cax2 = axs[1].imshow(depth_pred_masked, cmap='viridis', vmin=vmin, vmax=vmax)
            axs[1].set_title('depth_pred')
            fig.colorbar(cax2, ax=axs[1])

            eps = 1e-5
            vmin, vmax = min(depth_diff_masked.min(),-eps), max(depth_diff_masked.max(), eps)
            vcenter = 0

            cax3 = axs[2].imshow(depth_diff_masked, cmap='coolwarm', norm=mpl.colors.TwoSlopeNorm(vcenter, vmin, vmax))

            # max_abs = np.abs(depth_diff_masked).max()
            # vmin, vmax = -max_abs, max_abs
            # cax3 = axs[2].imshow(depth_diff_masked, cmap='coolwarm', vmin=vmin, vmax=vmax)

            axs[2].set_title('depth_diff')
            fig.colorbar(cax3, ax=axs[2])

            fig.savefig(file_path)
            plt.close(fig)




        if geometry_analysis_type != "unspecified" and geometry_analysis_type != None: 
            sampling_width = 0.5

            orig_near = self.model.config.near_plane
            orig_far = self.model.config.far_plane
            self.model.config.near_plane = 0
            self.model.config.far_plane = 2*sampling_width*self.datamanager.train_dataparser_outputs.dataparser_scale
            
            if geometry_analysis_type == "plane":
                plane_dimensions = geometry_analysis_dimensions.get("size", [1,1])
                camera_ray_bundle = plane_eval_ray_bundle(self.datamanager.train_dataparser_outputs, sampling_width, dimensions=plane_dimensions).to(self.device)
            elif geometry_analysis_type == "sphere":
                plane_dimensions = (1,1)
                sphere_radius = geometry_analysis_dimensions.get("radius", 0.5)
                camera_ray_bundle = sphere_eval_ray_bundle(self.datamanager.train_dataparser_outputs, sampling_width, radius=sphere_radius).to(self.device)
            else:
                raise Exception(f"unknown geometry_analysis_type: {geometry_analysis_type}")
            outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

            self.model.config.near_plane = orig_near
            self.model.config.far_plane = orig_far
            if not "rgb" in outputs:
                return metrics_dict

            rgb = outputs["rgb"] if "rgb" in outputs else outputs["rgb_fine"]
            acc = outputs["accumulation"] if "accumulation" in outputs else outputs["accumulation_fine"]
            rgb = torch.concat([rgb, acc], dim=-1)
            depth = outputs["depth"] / self.datamanager.train_dataparser_outputs.dataparser_scale
            mask = depth < 2 * sampling_width
            # mask = torch.abs(depth - torch.mean(depth)) < 1 * torch.std(depth)
            acc = colormaps.apply_colormap(acc)
            depth_vis = torch.clone(depth)
            depth_vis[torch.logical_not(mask)] = torch.min(depth[mask]) if depth[mask].numel() > 0 else 0
            depth_vis = colormaps.apply_depth_colormap(
                depth_vis,
                accumulation=acc,
            )
            depth_vis = torch.concat([depth_vis, mask], dim=-1)
            z = (sampling_width - depth)
            depth_diff = -z
            depth_gt = torch.ones_like(depth)*sampling_width

            metrics_dict["max_surface_diff"] = float(torch.max(z))
            metrics_dict["min_surface_diff"] = float(torch.min(z))
            metrics_dict["std_surface_diff"] = float(torch.std(z))
            metrics_dict["mean_surface_diff"] = float(torch.mean(z))

            if output_path is not None:
                save_as_image(rgb, output_path / "rgb.png")
                save_as_image(acc, output_path / "acc.png")
                save_as_image(depth_vis, output_path / "depth.png")
                # save_weight_distribution_plot(output_path / f"weight_hist.png", outputs, [0, 2 * sampling_width], self.datamanager.train_dataparser_outputs.dataparser_scale, plot_cdf = False)
                save_weight_distribution_plot(output_path / f"weight_cfd.png", outputs, [0, 2 * sampling_width], self.datamanager.train_dataparser_outputs.dataparser_scale, plot_cdf = True)
                save_depth_vis(output_path / "depth_plot.png", depth_gt, depth, depth_diff, mask)

                # Convert PyTorch tensor to NumPy array
                z_numpy = z.squeeze().cpu().numpy()
                np.save(output_path / "z.npy", z_numpy)

                origins = camera_ray_bundle.origins.cpu()
                directions = camera_ray_bundle.directions.cpu()
                surface = origins + directions * outputs["depth"].cpu()
                surface = convert_from_transformed_space(surface, self.datamanager.train_dataparser_outputs)

                # Create the 3D plot
                fig = plt.figure()
                fig.suptitle(method_name)
                ax = fig.add_subplot(111, projection='3d')

                # Setting the TwoSlopeNorm
                eps = 1e-5
                vmin, vmax = min(z_numpy.min(), -eps), max(z_numpy.max(), eps)
                vcenter = 0
                norm = mpl.colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
                surface = ax.plot_surface(surface[:,:,0], surface[:,:,1], surface[:,:,2], rstride=10, cstride=10, linewidth=0, facecolors=plt.cm.coolwarm(norm(z_numpy)))

                # mappable = plt.cm.ScalarMappable(norm=plt.Normalize(z_numpy.min(), z_numpy.max()), cmap=plt.cm.jet)
                # mappable.set_array(z_numpy)
                # plt.colorbar(mappable, ax=ax)


                # surface = ax.plot_surface(surface[:,:,0], surface[:,:,1], surface[:,:,2], rstride=10, cstride=10, cmap='viridis')

                # Labels and title
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('3D Surface Plot of Z')

                # Function to update the plot at each frame
                def update(frame):
                    ax.view_init(elev=20., azim=3.6*frame)
                    return surface,

                # Create animation
                ani = FuncAnimation(fig, update, frames=np.arange(0, 100), blit=False, repeat=False)

                # To save the animation
                ani.save(output_path / '3D_rotation.gif', writer='imagemagick')
                plt.close(fig)


        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        depth_silog = np.zeros(num_images, np.float32)
        depth_log10 = np.zeros(num_images, np.float32)
        depth_rms = np.zeros(num_images, np.float32)
        depth_log_rms = np.zeros(num_images, np.float32)
        depth_abs_rel = np.zeros(num_images, np.float32)
        depth_sq_rel = np.zeros(num_images, np.float32)
        depth_d1 = np.zeros(num_images, np.float32)
        depth_d2 = np.zeros(num_images, np.float32)
        depth_d3 = np.zeros(num_images, np.float32)


        for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
            image_idx = batch["image_idx"]
            depth_filepath = depth_filenames[image_idx]
            depth_gt = read_depth_map(str(depth_filepath), self.device)
            mask = depth_gt <= 2 / self.datamanager.train_dataparser_outputs.dataparser_scale
            depth_gt[torch.logical_not(mask)] = torch.min(depth_gt[mask])
            outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            _metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
            metrics_dict = { **metrics_dict, **_metrics_dict }
            if not "rgb" in outputs:
                continue
            rgb_pred = outputs["rgb"].cpu()
            acc = outputs["accumulation"] if "accumulation" in outputs else outputs["accumulation_fine"]

            rgb_pred_loss, rgb_gt_loss = self.model.renderer_rgb.blend_background_for_loss_computation(
                pred_image=outputs["rgb"],
                pred_accumulation=acc,
                gt_image=batch["image"],
            )
            rgb_pred_loss, rgb_gt_loss = rgb_pred_loss.cpu(), rgb_gt_loss.cpu()
            rgb_pred = torch.concat([rgb_pred, torch.ones((rgb_pred.shape[0], rgb_pred.shape[1], 1))], dim=-1)
            # rgb_pred = torch.concat([rgb_pred, 1 - acc.cpu()], dim=-1)
            rgb_gt = batch["image"].cpu()
            
            rgb_compare = torch.concat([rgb_gt, rgb_pred], dim=1)
            rgb_compare_loss = torch.concat([rgb_gt_loss, rgb_pred_loss], dim=1)
            acc = acc
            acc_vis = colormaps.apply_colormap(acc)

            depth_pred = outputs["depth"] / self.datamanager.train_dataparser_outputs.dataparser_scale

            depth_pred_vis = colormaps.apply_depth_colormap(
                depth_pred,
                accumulation=outputs["accumulation"],
            )
            depth_gt_vis = colormaps.apply_depth_colormap(
                depth_gt,
            )
            depth_gt_vis = torch.concat([depth_gt_vis, mask], dim=-1)
            depth_diff = depth_pred - depth_gt
            depth_diff[depth_gt > 1000] = torch.min(depth_diff[mask])
            depth_diff_vis = colormaps.apply_depth_colormap(
                depth_diff,
            )
            depth_diff_vis = torch.concat([depth_diff_vis, mask], dim=-1)

            if output_path is not None:
                # save_as_image(rgb_pred, output_path / f"rgb_pred_{image_idx:04d}.png")
                # save_as_image(rgb_gt, output_path / f"rgb_gt_{image_idx:04d}.png")
                # save_as_image(rgb_pred_loss, output_path / f"rgb_pred_loss_{image_idx:04d}.png")
                # save_as_image(rgb_gt_loss, output_path / f"rgb_gt_loss_{image_idx:04d}.png")
                save_as_image(rgb_compare, output_path / f"rgb_compare_{image_idx:04d}.png")
                save_as_image(rgb_compare_loss, output_path / f"rgb_compare_loss_{image_idx:04d}.png")
                save_as_image(acc_vis, output_path / f"acc_{image_idx:04d}.png")
                # save_as_image(depth_pred_vis, output_path / f"depth_pred_{image_idx:04d}.png")
                # save_as_image(depth_gt_vis, output_path / f"depth_gt_{image_idx:04d}.png")
                # save_as_image(depth_diff_vis, output_path / f"depth_diff_{image_idx:04d}.png")
                save_depth_vis(output_path / f"depth_plot_{image_idx:04d}.png", depth_gt, depth_pred, depth_diff, mask)
                # save_weight_distribution_plot(output_path / f"weight_hist_{image_idx:04d}.png", outputs, [0, 6], self.datamanager.train_dataparser_outputs.dataparser_scale, plot_cdf = False)



            depth_silog[image_idx], depth_log10[image_idx], depth_abs_rel[image_idx], depth_sq_rel[image_idx], depth_rms[image_idx], depth_log_rms[image_idx], depth_d1[image_idx], depth_d2[image_idx], depth_d3[image_idx] = compute_errors(
                    depth_gt[mask].to("cpu").numpy(), depth_pred[mask].to("cpu").numpy())
            
        metrics_dict["depth_silog"] = float(depth_silog.mean())
        metrics_dict["depth_log10"] = float(depth_log10.mean())
        metrics_dict["depth_abs_rel"] = float(depth_abs_rel.mean())
        metrics_dict["depth_sq_rel"] = float(depth_sq_rel.mean())
        metrics_dict["depth_rms"] = float(depth_rms.mean())
        metrics_dict["depth_log_rms"] = float(depth_log_rms.mean())
        metrics_dict["depth_d1"] = float(depth_d1.mean())
        metrics_dict["depth_d2"] = float(depth_d2.mean())
        metrics_dict["depth_d3"] = float(depth_d3.mean())

        self.train()
        return metrics_dict