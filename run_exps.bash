DATASET_DIR=~/Datasets/NeRF/nerf-geometry-analysis
DATASET="$1"
# FAST_METHODS=("instant-ngp" "tensorf")
# SLOW_METHODS=("vanilla-nerf" "neus" "mipnerf")
# METHODS=("instant-ngp" "kplanes" "instant-ngp-bounded" "tensorf" "nerfacto")
# METHODS=("tensorf" "instant-ngp" "kplanes" "instant-ngp-bounded" "nerfacto")
METHODS=("instant-ngp" "tensorf" "nerfacto")
# METHODS=("tensorf" "instant-ngp" "kplanes" "nerfacto")
# DATASETS=($(ls -d "$DATASET_DIR"/*))
DATASETS=("pattern_plane1" "checkered_plane" "black_line_bg_white" "black_square_bg_white")
DATASETS=("$DATASET_DIR/${DATASETS[@]}")

get_most_recent_dir() {
  dir_path="$1"
  most_recent=$(ls -td "${dir_path}"/* | head -n 1)
  echo $most_recent
}

eval () {
    render_dir="$1/renders"
    rm -rf $render_dir
    bash eval.bash $1
    convert -delay 100 -loop 0 -background red -alpha remove -alpha off "$render_dir"/rgb_compare_*.png "$render_dir"/rgb_compare.gif || true
    convert -delay 100 -loop 0 -background red -alpha remove -alpha off "$render_dir"/depth_plot_*.png "$render_dir"/depth_plot.gif || true
    convert -delay 10 -loop 0 -background white -alpha remove -alpha off "$render_dir"/line_contour_*.jpeg "$render_dir"/line_contour.gif || true
    convert -delay 100 -loop 0 -background red -alpha remove -alpha off "$render_dir"/contour_rgb_*.png "$render_dir"/contour_rgb.gif || true
}


# FARS=(1 2 3 4 5 6 7 8)

# for FAR in "${FARS[@]}"; do
#   for METHOD in "${METHODS[@]}"; do
#     # DATASET=~/Datasets/NeRF/nerf-geometry-analysis/pattern_plane1
#     ns-train nga-$METHOD --vis "tensorboard" --viewer.quit-on-train-completion True --data $DATASET --pipeline.model.far_plane $FAR

#     dataset_name=$(basename "$DATASET")
#     output_dir=$(get_most_recent_dir "outputs/$dataset_name/nga-$METHOD")
#     eval $output_dir
#   done
# done


for METHOD in "${METHODS[@]}"; do
  # DATASET=~/Datasets/NeRF/nerf-geometry-analysis/pattern_plane1
  ns-train nga-$METHOD --vis "tensorboard" --viewer.quit-on-train-completion True --data $DATASET 

  dataset_name=$(basename "$DATASET")
  output_dir=$(get_most_recent_dir "outputs/$dataset_name/nga-$METHOD")
  eval $output_dir
done