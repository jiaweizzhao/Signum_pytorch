#coding=utf-8
import os
import random
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class Food(data.Dataset):    #1400 700

    def __init__(self, root, transforms=None, train=True, test=False):
        '''
        划分数据集，训练，测试
        test为True,返回测试集，30%
        train为True,返回训练集，70%
        :param root:
        :param transforms:
        :param train:
        :param test:
        '''
        self.test = test
        second_root = []
        imgs = []

        root = os.path.join(root,'food-101','images')


        for i, class_image in zip(range(2), os.listdir(root)):
            second_root.append(os.path.join(root,class_image))
            imgs = imgs + [os.path.join(second_root[i], img)for img in os.listdir(second_root[i])]

        num_imgs = len(imgs)
        random.seed(100)
        random.shuffle(imgs)
        self.imgs = imgs
        if self.test:
            self.imgs = imgs[int(0.7*num_imgs):]
        else:
            self.imgs = imgs[:int(0.7*num_imgs)]

        if transforms is None:

            normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        '''
        返回一张图片
        :param self:
        :param index:
        :return:
        '''
        img_path = self.imgs[index]
        data = Image.open(img_path)
        #print(self.test)
        #print(img_path)
        label_name = img_path.split('\\')[-2]
        if label_name == 'apple_pie':
            label = 0
        elif label_name == 'baby_back_ribs':
            label = 1
        #print(label)
        data = self.transforms(data)

        return data,label

    def __len__(self):
        '''
        返回该类的所有图片数量
        :return:
        '''
        return len(self.imgs)