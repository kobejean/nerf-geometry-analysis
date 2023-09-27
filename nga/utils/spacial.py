import torch
# from torch.nn.functional import normalize
import numpy as np

def convert_to_transformed_space(x, dataparser_outputs, is_direction=False):
    scale = dataparser_outputs.dataparser_scale
    transform = dataparser_outputs.dataparser_transform
    x = torch.matmul(x, transform[:,0:3].transpose(0,1))
    if not is_direction:
        x += transform[:,3]
    x *= scale
    return x

def convert_from_transformed_space(x, dataparser_outputs, is_direction=False):
    scale = dataparser_outputs.dataparser_scale
    transform = dataparser_outputs.dataparser_transform
    x /= scale
    if not is_direction:
        x -= transform[:,3]
    R = torch.linalg.inv(transform[:,0:3])
    x = torch.matmul(x, R.transpose(0,1))
    return x