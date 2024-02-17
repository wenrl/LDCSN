import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import random

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return
    channels = img.shape[2]
    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha
    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])
    return Image.fromarray(img)


class Dataset_align(data.Dataset):

    def __init__(self, root, data_list_file, mask=False, input_shape=(3, 112, 112)):
        self.root = root
        self.random = 3
        self.mask = mask
        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()
        imgs = [os.path.join(root, img) for img in imgs]

        self.imgs = imgs
        normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


        self.transforms = T.Compose([
                # T.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                # T.RandomCrop(self.input_shape[1:]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        length = 224
        sample = self.imgs[index]

        splits = sample.split()
        img_path = splits[0]
        occ_path = img_path.replace('casia_align','casia_align__masked')
        if self.mask == True:
            if os.path.isfile(occ_path):
                if random.choice(range(10)) > 5:
                    data = Image.open(occ_path).convert('RGB')
                else:
                    data = Image.open(img_path).convert('RGB')
            else:
                data = Image.open(img_path).convert('RGB')
        else:
            data = Image.open(img_path).convert('RGB')

        data = data.resize((length,length))
        data = self.transforms(data)
        label = np.long(img_path.split('/')[-2])

        return data.float(), label

    def __len__(self):
        return len(self.imgs)
class Dataset_VGG_align(data.Dataset):

    def __init__(self, root, data_list_file, input_shape=(3, 112, 112)):

        self.input_shape = input_shape
        self.root = root
        self.random = 3

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()
        imgs = [os.path.join(root, img) for img in imgs]
        self.imgs = imgs
        normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


        self.transforms = T.Compose([
                # T.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                # T.RandomCrop(self.input_shape[1:]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):

        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        data = Image.open(img_path).convert('RGB')
        data = data.resize((112,112))
        data = self.transforms(data)
        label = np.long(img_path.split('/')[-2])

        return data.float(), label

    def __len__(self):
        return len(self.imgs)


class Dataset(data.Dataset):

    def __init__(self, root, data_list_file, mask=False, input_shape=(3, 112, 112)):

        self.input_shape = input_shape
        self.root = root
        self.random = 3
        self.mask = mask
        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()
        imgs = [os.path.join(root, img) for img in imgs]
        self.imgs = imgs
        normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        self.transforms = T.Compose([
                # T.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                # T.RandomCrop(self.input_shape[1:]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        occ_path = img_path.replace('jpg','png')
        occ_path = occ_path.replace('CASIA-WebFace','CASIA-WebFace_masks')
        if self.mask == True:
            if os.path.isfile(occ_path):
                if random.choice(range(3)) > 0:
                    data = cv2.imread(img_path)
                    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                    mask_image = cv2.imread(occ_path, cv2.IMREAD_UNCHANGED)
                    data = overlay_image_alpha(data,
                            mask_image[:, :, 0:3],
                            (0, 0),
                            mask_image[:, :, 3] / 255.0)
                else:
                    data = Image.open(img_path).convert('RGB')
            else:
                data = Image.open(img_path).convert('RGB')


        else:
            data = Image.open(img_path).convert('RGB')

        data = data.resize((112,112))
        data = self.transforms(data)
        label = np.long(img_path.split('/')[-2])

        return data.float(), label

    def __len__(self):
        return len(self.imgs)

