#coding=utf-8

import torch

from pre_process import find_center
import torch.nn as nn
from torchvision import  models, transforms


import math

def load_info():
    BATCH_SIZE = 25
    NUM_EPOCHS = 200
    BATCHS_EPOCHS = 1
    INIT_LR = 0.01
    MOMENTUM = 0.9
    LR_DECAY_EPOCH = 20
    SELECTION_TYPE = 'RANDOM'
    COMBINE_PAR = 2

    return BATCH_SIZE,NUM_EPOCHS,BATCHS_EPOCHS,INIT_LR,MOMENTUM,LR_DECAY_EPOCH,SELECTION_TYPE,COMBINE_PAR