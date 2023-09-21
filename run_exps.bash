DATASET_DIR=~/Datasets/NeRF/nerf-geometry-analysis
# FAST_METHODS=("instant-ngp" "tensorf")
# SLOW_METHODS=("vanilla-nerf" "neus" "mipnerf")
# METHODS=("kplanes" "instant-ngp-bounded" "instant-ngp" "tensorf" "nerfacto")
METHODS=("tensorf" "instant-ngp" "nerfacto")
# DATASETS=($(ls -d "$DATASET_DIR"/*))
DATASETS=("pattern_plane1" "checkered_plane" "black_line_bg_white" "black_square_bg_white")
DATASETS=("$DATASET_DIR/${DATASETS[@]}")

# for METHOD in "${FAST_METHODS[@]}"; do
#   for DATASET in "${DATASETS[@]}"; do
#     echo "ns-train $METHOD --vis "viewer" --viewer.quit-on-train-completion True --data $DATASET nerfstudio-data --data $DATASET"
#     ns-train $METHOD --vis "viewer" --viewer.quit-on-train-completion True --data $DATASET nerfstudio-data --data $DATASET
#   done
# done

# for METHOD in "${SLOW_METHODS[@]}"; do
#   DATASET=~/Datasets/NeRF/nerf-geometry-analysis/pattern_plane1_720x480
#   echo "ns-train $METHOD --vis "viewer" --viewer.quit-on-train-completion True --data $DATASET nerfstudio-data --data $DATASET"
#   ns-train $METHOD --vis "viewer" --viewer.quit-on-train-completion True --data $DATASET nerfstudio-data --data $DATASET
# done

for METHOD in "${METHODS[@]}"; do
  DATASET=~/Datasets/NeRF/nerf-geometry-analysis/long_plane_2_cameras
  echo "ns-train $METHOD --vis "tensorboard" --viewer.quit-on-train-completion True --data $DATASET nerfstudio-data --data $DATASET"
  ns-train $METHOD --vis "tensorboard" --viewer.quit-on-train-completion True --data $DATASET nerfstudio-data --data $DATASET
done

# for METHOD in "${METHODS[@]}"; do
#   ns-train $METHOD --load-config configs/$METHOD/config.yml
# done