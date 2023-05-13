'''Calculate output dimension of the convolutional kernel!'''
import math
ninput = 3
padding = 0
settings = [
    [3, 1],
    [3, 2],
    [3, 3],
    [4, 2],
    [4, 3], # kernel_size, stride
    [5, 2],
    [5, 3],
    [7, 2],
    [7, 3],
    [7, 5],
    [8, 2],
    [8, 3],
    [8, 5],
    [10, 2],
    [10, 3],
    [12, 2],
    [12, 3],
    [16, 2],
    [16, 3],
    [20, 2],
    [20, 3],
    [50, 5],
    [50, 6],
    [64, 6]
]

for setting in settings:
    kernel_size = setting[0]
    stride = setting[1]
    noutput = (ninput+2*padding-kernel_size)/stride+1
    if noutput % 1 == 0 and noutput > 0:
        print(f'[{ninput} -> {int(noutput)}] kernel_size: {kernel_size}, stride = {stride}, padding = {padding}')