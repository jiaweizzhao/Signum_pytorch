#python 3.6

import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms import Compose, CenterCrop, Normalize

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)


def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')


class trainsets(Dataset):

    def __init__(self, root):
        root = os.path.join(root,'VOC2012')
        self.images_root = os.path.join(root, 'JPEGImages')
        self.labels_root = os.path.join(os.path.join(root, 'ImageSets'),'Main')

        filename = os.path.join(self.labels_root,'aeroplane_train.txt')
        
        file = open(filename)

        self.filenames = []
        self.labels = []

        while(True):
            line = file.readline()
            if not line:
                break
            line = line.split(' ')
            self.filenames.append(line[0])
            if line[1] == '' :
                a = int(line[2].strip('\n'))
            else:
                a = int(line[1].strip('\n'))
            if a == -1:
                a = 0
            self.labels.append(a)

        transform = Compose([
            CenterCrop(224),
            ToTensor(),
            Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        self.transform = transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        label = self.labels[index]

        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.filenames)    



class testsets(Dataset):

    def __init__(self, root):
        root = os.path.join(root,'VOC2012')
        self.images_root = os.path.join(root, 'JPEGImages')
        self.labels_root = os.path.join(os.path.join(root, 'ImageSets'),'Main')

        filename = os.path.join(self.labels_root,'aeroplane_val.txt')
        
        file = open(filename)

        self.filenames = []
        self.labels = []

        while(True):
            line = file.readline()
            if not line:
                break
            line = line.split(' ')
            self.filenames.append(line[0])
            if line[1] == '' :
                a = int(line[2].strip('\n'))
            else:
                a = int(line[1].strip('\n'))
            if a == -1:
                a = 0
            self.labels.append(a)

        transform = Compose([
            CenterCrop(224),
            ToTensor(),
            Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        self.transform = transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        label = self.labels[index]

        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.filenames) 