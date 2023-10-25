import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.special import sph_harm
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

def sph_harm_real(l, m, theta, phi):
    sign = 1 if m % 2 == 0 else -1
    if m < 0:
        Y = np.sqrt(2) * sign * sph_harm(np.abs(m), l, phi, theta).imag
    elif m == 0:
        Y = sph_harm(0, l, phi, theta).real
    else:
        Y = np.sqrt(2) * sign * sph_harm(m, l, phi, theta).real
    return Y

# Image dimensions
width, height = 4096, 2048

# Generate a grid of angles
theta, phi = np.mgrid[0:np.pi:height*1j, 0:2*np.pi:width*1j]

def normalize(Y):
    Y_spread = Y.max() - Y.min()
    return (Y - Y.min()) / Y_spread if Y_spread > 0 else Y - Y.min()

def save(filename, colors):
    plt.imsave(os.path.join(script_dir, filename), colors, origin='upper')

max_l = 20

def map_cmpx():
    intensity = np.abs(Y)
    hue = np.angle(Y, deg=True) / 360.0  # converting angle in degrees to [0, 1]

    # Convert hue and intensity to RGB
    colors = np.zeros((height, width, 3))
    colors[..., 0] = hue
    colors[..., 1] = intensity  # Saturation
    colors[..., 2] = intensity  # Value/Brightness
    colors_rgb = mcolors.hsv_to_rgb(colors)

for l in range(max_l+1):
    Y_l_real = sph_harm_real(l, -l, theta, phi)
    for m in range(-l+1,l+1):
        Y_lm_real = sph_harm_real(l, m, theta, phi)
        Y_l_real += Y_lm_real
        
    Y_l_real_normalized = normalize(Y_l_real)
    # colors = plt.cm.hsv(Y_l_real_normalized)
    # save(f"sh_{l}_real.png", colors)

    # Convert hue and intensity to RGB
    colors = np.zeros((height, width, 3))
    colors[..., 0] = Y_l_real_normalized
    colors[..., 1] = (1+np.flip(Y_l_real_normalized))/2  # Saturation
    colors[..., 2] = (1+np.flip(Y_l_real_normalized, axis=1))/2  # Value/Brightness
    print(l)
    colors = mcolors.hsv_to_rgb(colors)
    save(f"sh_{l}_real1.png", colors)

# for l in range(21):
#     Y_l_cmpx = sph_harm(np.abs(-l), l, phi, theta)
#     for m in range(0,l+1):
#         Y_lm_cmpx = sph_harm(m, l, phi, theta)
#         Y_l_cmpx += Y_lm_cmpx
        
#     intensity = np.abs(Y_l_cmpx)
#     intensity /= np.max(intensity)
#     angle = np.angle(Y_l_cmpx) 
#     hue = 0.5 + angle / (2*np.pi)  # converting angle in degrees to [0, 1]

#     # Convert hue and intensity to RGB
#     colors = np.zeros((height, width, 3))
#     colors[..., 0] = hue
#     colors[..., 1] = (3+np.cos(np.flip(angle)))/4 #(1+intensity)/2  # Saturation
#     colors[..., 2] = (3+np.sin(np.flip(angle, axis=1)))/4  # Value/Brightness
#     # colors[..., 1] = (1+np.cos(angle))/2 #(1+intensity)/2  # Saturation
#     # colors[..., 2] = (1+np.sin(angle))/2  # Value/Brightness
#     print(np.max(hue), np.min(hue))
#     colors = mcolors.hsv_to_rgb(colors)
#     save(f"sh_{l}_cmpx4.png", colors)