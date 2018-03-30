# coding=utf-8
import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torch.utils import data


class Caltech(data.Dataset):
    def __init__(self, root,PC,train=True, test=False):

        root = os.path.join(root,'256_ObjectCategories')

        self.PC = PC
        self.train = train
        self.test = test
        labels = []
        imgs = []
        # class_path=[D:\256_ObjectCategories\001.ak47,...]
        class_path = [os.path.join(root, class_i) for class_i in os.listdir(root)]

        for class_path_i in class_path:
            imgs = imgs + [os.path.join(class_path_i, img) for img in os.listdir(class_path_i)]
            class_path_i_num = len([img_i for img_i in os.listdir(class_path_i)])  # 该类下图片数量
            if self.PC == 207:
                labels = labels + [int(class_path_i.split('/')[-1].split('.')[-2]) -1] * class_path_i_num
            else:
                labels = labels + [int(class_path_i.split('\\')[-1].split('.')[-2]) -1] * class_path_i_num

        num_imgs = len(imgs)
        random.seed(100)
        random.shuffle(imgs)
        num_labels = len(labels)
        random.seed(100)
        random.shuffle(labels)

        if self.train:
            self.imgs = imgs[:int(0.7 * num_imgs)]
            self.labels = labels[:int(0.7 * num_labels)]
        elif self.test:
            self.imgs = imgs[int(0.7 * num_imgs):]
            self.labels = labels[int(0.7 * num_labels)]

        normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        self.transforms = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])

    def __getitem__(self, index):
        # D:\256_ObjectCategories\001.ak47\001_0001.jpg
        img_path = self.imgs[index]
        if self.PC == 207:
            label = int(img_path.split('/')[-1].split('_')[-2]) - 1
            path_num_b = img_path.split('/')[-1].split('.')[-2]
            path_num_a = img_path.split('/')[-2]
            path_num = path_num_a + '/' + path_num_b
        else:
            label = int(img_path.split('\\')[-1].split('_')[-2]) - 1
            path_num_b = img_path.split('\\')[-1].split('.')[-2]
            path_num_a = img_path.split('\\')[-2]
            path_num = path_num_a + '\\' + path_num_b

        data = Image.open(img_path).convert('RGB')
        data = self.transforms(data)

        assert label >=0
        assert  label <=255

        return data, label

    def __len__(self):
        return len(self.imgs)