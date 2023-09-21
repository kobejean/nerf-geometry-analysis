## Install Deps
```
sudo apt install libopenexr-dev zlib1g-dev
pip install git+https://github.com/jamesbowman/openexrpython.git
pip install git+https://github.com/kobejean/kplanes_nerfstudio.git
```

## Blender file

The blender project file is located at: `<repo-root>/Blender/checkerboard.blend`


## Blender Add-On Installation

1. Open Blender (3.0.0 or above)
1. In Blender, head to **Edit > Preferences > Add-ons**, and click **Install...**
1. Select the **ZIP** file in `<repo-root>/Blender/Add-ons/NeRFDataset.zip`, and activate the add-on (**Object: NeRFDataset**)


## How to use 

The add-on properties panel is available under `3D View > N panel > NeRFDataset` (the **N panel** is accessible under the 3D viewport when pressing `N`).

## Addon Development Workflow

- Find the location where Blender stores add-ons. It's usually something like:
On Windows: `%USERPROFILE%\AppData\Roaming\Blender Foundation\Blender\<version>\scripts\addons`
On macOS: `/Applications/Blender.app/Contents/Resources/3.5/scripts/addons/NeRFDataset`
On Linux: `~/.config/blender/<version>/scripts/addons`
- Create a symbolic link to your development directory in this location. The exact command will vary by operating system:
On Windows: mklink /D Link Target
On macOS/Linux: ln -s Target Link

linux:
```
ln -s ~/Code/nerf-geometry-analysis/Blender/Add-ons/NeRFDataset ~/.config/blender/3.0/scripts/addons/NeRFDataset
```

macOS:
```
ln -s ~/Developer/GitHub/nerf-geometry-analysis/Blender/Add-ons/NeRFDataset /Applications/Blender.app/Contents/Resources/3.5/scripts/addons/NeRFDataset
```
