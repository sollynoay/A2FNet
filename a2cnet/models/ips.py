import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_unshuffle_height(input, downscale_factor):
    '''
    input: batchSize * c * w * k*h
    kdownscale_factor: k
    batchSize * c * w * k*h -> batchSize * k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * c,
                               1, downscale_factor, 1],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(1):
            kernel[x + y, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=(downscale_factor,1), groups=c)

def pixel_unshuffle_width(input, downscale_factor):
    '''
    input: batchSize * c * w * k*h
    kdownscale_factor: k
    batchSize * c * w * k*h -> batchSize * k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * c,
                               1, 1, downscale_factor],
                         device=input.device)
    for y in range(1):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=(1,downscale_factor), groups=c)

def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor,  groups=c)
