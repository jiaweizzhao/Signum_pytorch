#coding=utf-8

import torch

from pre_process import find_center
import torch.nn as nn
from torchvision import  models, transforms


import math

def load_frozen_num():
    return 44

def load_register_hook(model,hook_1,hook_2):
    handle1 = model.layer4[0].downsample[1].register_forward_hook(hook_1)  # (1000, 64, 56, 56)
    handle2 = model.avgpool.register_forward_hook(hook_2)  # (1000, 512, 7, 7)

    return handle1,handle2

def load_multi_register_hook(model,hook_1,hook_2,hook_3,hook_4):
    handle1 = model.features[20].register_forward_hook(hook_1)
    handle2 = model.classifier[5].register_forward_hook(hook_2)
    handle3 = model.classifier[5].register_forward_hook(hook_2)
    handle4 = model.classifier[5].register_forward_hook(hook_2)

    return handle1,handle2,handle3,handle4


def load_model(CLASS_NUM):
    model_ft = models.resnet18(pretrained=True)
    model_ft.fc = nn.Linear(in_features=512, out_features=CLASS_NUM)

    return model_ft