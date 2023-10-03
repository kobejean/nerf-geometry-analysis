
eval () {
    render_dir="$1/renders"
    rm -rf $render_dir
    bash eval.bash $1
    convert -delay 100 -loop 0 -background red -alpha remove -alpha off "$render_dir"/contour_rgb_*.png "$render_dir"/contour_rgb.gif
    convert -delay 100 -loop 0 -background red -alpha remove -alpha off "$render_dir"/rgb_compare_*.png "$render_dir"/rgb_compare.gif
    convert -delay 100 -loop 0 -background red -alpha remove -alpha off "$render_dir"/depth_plot_*.png "$render_dir"/depth_plot.gif
}



eval outputs/checkered_plane/nga-instant-ngp/2023-09-30_115217
# eval outputs/checkered_plane/nga-nerfacto/2023-09-30_120141
# eval outputs/checkered_plane/nga-tensorf/2023-09-30_111855

# eval outputs/checkered_sphere_specular/nga-instant-ngp/2023-09-28_081912
# eval outputs/checkered_sphere_specular/nga-nerfacto/2023-09-28_080744
# eval outputs/checkered_sphere_specular/nga-tensorf/2023-09-28_073815

eval outputs/checkered_sphere/nga-instant-ngp/2023-09-27_224714
# eval outputs/checkered_sphere/nga-nerfacto/2023-09-27_223556
# eval outputs/checkered_sphere/nga-tensorf/2023-09-27_220653
# eval outputs/checkered_sphere/nga-vanilla-nerf/2023-09-25_222035

eval outputs/checkered_cube/nga-instant-ngp/2023-09-30_140921
# eval outputs/checkered_cube/nga-nerfacto/2023-09-30_141605
# eval outputs/checkered_cube/nga-tensorf/2023-09-30_134549

# eval outputs/pattern_plane1/nga-instant-ngp/2023-09-27_233353
# eval outputs/pattern_plane1/nga-nerfacto/2023-09-27_232236
# eval outputs/pattern_plane1/nga-tensorf/2023-09-27_225655
# eval outputs/pattern_plane1/nga-instant-ngp/2023-09-27_233353

# eval outputs/checkered_sphere/nga-vanilla-nerf/2023-09-25_222035
