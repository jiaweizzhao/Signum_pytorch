#coding=utf-8

import torch

from pre_process import find_center

import math

def load_info():
    datasets_dir = 'D:\Jiawei\ExperimentADMA\datasets'
    TEST_BATCH_SIZE = 400
    TEST_BATCHS = 10

    return  datasets_dir,TEST_BATCH_SIZE,TEST_BATCHS