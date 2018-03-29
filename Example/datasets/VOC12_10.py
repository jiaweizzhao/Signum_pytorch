#python 3.6

import numpy as np
import os
import random
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)


def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')


class return_datasets(Dataset):
    def __init__(self,datasets,images_root):
        self.datasets = datasets
        self.transform = Compose([
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.images_root = images_root

    def __getitem__(self, index):

        filename = self.datasets[index][0]
        label = self.datasets[index][1]
        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        image = self.transform(image)

        assert label <= 9
        assert label >= 0

        return image, label

    def __len__(self):
        return len(self.datasets)



def datasets(root):
    root = os.path.join(root, 'VOC2012')
    images_root = os.path.join(root, 'JPEGImages')
    labels_root = os.path.join(os.path.join(root, 'ImageSets'), 'Main')

    second_path = []
    filenames = []
    labels = []

    class_num = 0
    for class_image in os.listdir(labels_root):
        if class_num == 10:
            break
        line = str(class_image).split('_')
        if line[1] != 'trainval.txt':
            pass
        else:
            second_path.append([os.path.join(labels_root, class_image),class_num])
            class_num += 1

    for filename,class_num in second_path:
        file = open(filename)

        while(True):
            line = file.readline()
            if not line:
                break
            line = line.split(' ')
            if line[1] == '':
                a = int(line[2].strip('\n'))
            else:
                a = int(line[1].strip('\n'))
            if a == 1:
                filenames.append(line[0])
                a  = class_num
                labels.append(a)

    datasets = list(zip(filenames,labels))
    random.shuffle(datasets)

    trainset = datasets[0:int(0.7 * len(datasets))]
    testset  = datasets[int(0.7 * len(datasets)):len(datasets)]

    return return_datasets(trainset,images_root) , return_datasets(testset,images_root)



# class testsets(Dataset):
#
#     def __init__(self, root):
#         root = os.path.join(root,'VOC2012')
#         self.images_root = os.path.join(root, 'JPEGImages')
#         self.labels_root = os.path.join(os.path.join(root, 'ImageSets'), 'Main')
#
#         filename = os.path.join(self.labels_root, 'aeroplane_val.txt')
#
#         file = open(filename)
#
#         self.filenames = []
#         self.labels = []
#         self.select_labels = []
#
#         while(True):
#             line = file.readline()
#             if not line:
#                 break
#             line = line.split(' ')
#             self.filenames.append(line[0])
#             if line[1] == '' :
#                 a = int(line[2].strip('\n'))
#             else:
#                 a = int(line[1].strip('\n'))
#             if a == -1:
#                 a = 0
#             self.labels.append(a)
#
#         self.select_labels = self.labels[0:10]
#
#         transform = Compose([
#             CenterCrop(224),
#             ToTensor(),
#             Normalize([.485, .456, .406], [.229, .224, .225]),
#         ])
#
#         self.transform = transform
#
#     def __getitem__(self, index):
#         if train==True:
#             filename = self.train_filenames[index]
#         elif test==True:
#             filename = self.test_filenames[index]
#
#         with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
#             image = load_image(f).convert('RGB')
#         if train == True:
#             label = self.train_labels[index]
#         elif test == True:
#             label = self.test_labels[index]
#         image = self.transform(image)
#
#         return image, label
#
#     def __len__(self):
#         return len(self.filenames)