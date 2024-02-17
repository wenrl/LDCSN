import os
import six
# import lmdb
import torch
import pickle
import random
import cv2
import numpy as np
import pyarrow as pa
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFile, ImageDraw
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter

import core.utils as utils
from core.config import config
import pdb

Occluders = r'C:\Users\86137\Desktop\task\data\datasets\occluder/'
Occluders_List = r'C:\Users\86137\Desktop\task\data\datasets\occluder/occluder.txt'
ImageFile.LOAD_TRUNCATED_IAMGES = True
class RWMFD(Dataset):
    def __init__(self, config, transform=None):
        super(RWMFD, self).__init__()
        self.RWMFD_path = r'C:\Users\86137\Desktop\task\data\datasets\RWMFD\RWMFD_part_1/'
        self.RWMFD_occ_path = r'C:\Users\86137\Desktop\task\data\datasets\RWMFD\RWMFD_part_1/'
        self.num_class = 22
        self.mode = config.TEST.MODE
        self.pairs = self.get_pairs_lines(r'C:\Users\86137\Desktop\task\data\datasets\RWMFD\RWMFD_pairs.txt')

        self.transform = transform
        self.valid_check()
        self.num_pairs = 100

    def get_pairs_lines(self, path):
        with open(path) as f:
            pairs_lines = f.readlines()[:]
        return pairs_lines

    def valid_check(self):
        valid_pairs = []
        for pair in self.pairs:
            p = pair.replace('\n', '').split('\t')

            if 3 == len(p):
                sameflag = 1
                name1 = '{:04}'.format(int(p[0])) + '/' + '{:04}.jpg'.format(int(p[1]))
                name2 = '{:04}'.format(int(p[0])) + '/' + '{:04}.jpg'.format(int(p[2]))#_surgical
            elif 4 == len(p):
                sameflag = 0
                name1 = '{:04}'.format(int(p[0])) + '/' + '{:04}.jpg'.format(int(p[1]))
                name2 = '{:04}'.format(int(p[2])) + '/' + '{:04}.jpg'.format(int(p[3]))#_surgical
            else:
                raise ValueError("WRONG LINE IN 'pairs.txt! ")

            if os.path.exists(self.RWMFD_path + name2):
                valid_pairs.append(p)

        print('valid rate: {:.4f}'.format(len(valid_pairs) / len(self.pairs)))

        self.pairs = valid_pairs

    def __getitem__(self, index):

        p = self.pairs[index]

        if 3 == len(p):
            sameflag = 1
            name1 = '{:04}'.format(int(p[0])) + '/' + '{:04}.jpg'.format(int(p[1]))
            name2 = '{:04}'.format(int(p[0])) + '/' + '{:04}.jpg'.format(int(p[2]))
            name3 = '{:04}'.format(int(p[0])) + '/' + '{:04}.jpg'.format(int(p[2]))
        elif 4 == len(p):
            sameflag = 0
            name1 = '{:04}'.format(int(p[0])) + '/' + '{:04}.jpg'.format(int(p[1]))
            name2 = '{:04}'.format(int(p[2])) + '/' + '{:04}.jpg'.format(int(p[3]))
            name3 = '{:04}'.format(int(p[2])) + '/' + '{:04}.jpg'.format(int(p[3]))

        else:
            raise ValueError("WRONG LINE IN 'pairs.txt! ")
        length = 112
        if os.path.isfile(self.RWMFD_path + name1):
            image = Image.open(self.RWMFD_path + name1).convert('RGB')
        else:
            image = Image.open(self.RWMFD_path + name1).convert('RGB')
        img1 = image.resize((length, length))
        image = Image.open(self.RWMFD_path + name2).convert('RGB')
        img2 = image.resize((length, length))
        image = Image.open(self.RWMFD_path + name3).convert('RGB')
        img2_occ = image.resize((length, length))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img2_occ = self.transform(img2_occ)
        return img1, img2, img2_occ, sameflag

    def __len__(self):
        return len(self.pairs)

class CFP(Dataset):
    def __init__(self, config, transform=None):
        super(CFP, self).__init__()
        self.cfp_path = config.DATASET.CFP_PATH
        self.cfp_occ_path = config.DATASET.CFP_OCC_PATH
        self.num_class = config.DATASET.CFP_CLASS
        self.mode = config.TEST.MODE
        self.pairs = self.get_pairs_lines(config.DATASET.CFP_PAIRS)

        self.transform = transform
        self.valid_check()
        self.num_pairs = len(self.pairs)

    def get_pairs_lines(self, path):
        with open(path) as f:
            pairs_lines = f.readlines()[:]
        return pairs_lines

    def valid_check(self):
        valid_pairs = []
        for pair in self.pairs:
            p = pair.replace('\n', '').split(',')

            if 2 == len(p):

                name1 = '{:03}'.format((int(p[0])-1)//10+1) + '/' + '{:02}.jpg'.format((int(p[0])-1)%10+1)
                name2 = '{:03}'.format((int(p[1])-1)//10+1) + '/' + '{:02}.jpg'.format((int(p[1])-1)%10+1)  # _surgical

                if abs(int(p[0])//10-int(p[1])//10) == 0:
                    sameflag = 1
                else:
                    sameflag = 0

            else:

                raise ValueError("WRONG LINE IN 'pairs.txt! ")


            if os.path.exists(self.cfp_path + name2) & os.path.exists(self.cfp_path + name1):
                valid_pairs.append(p)

        print('valid rate: {:.4f}'.format(len(valid_pairs) / len(self.pairs)))

        self.pairs = valid_pairs

    def __getitem__(self, index):

        p = self.pairs[index]

        if 2 == len(p):
            name1 = '{:03}'.format((int(p[0])-1)//10+1) + '/' + '{:02}.jpg'.format((int(p[0])-1)%10+1)
            name2 = '{:03}'.format((int(p[1])-1)//10+1) + '/' + '{:02}.jpg'.format((int(p[1])-1)%10+1)  # _surgical
            name3 = name2
            if abs((int(p[0])-1) // 10 - (int(p[1])-1) // 10) == 0:
                sameflag = 1
            else:
                sameflag = 0

        else:
            raise ValueError("WRONG LINE IN 'pairs.txt! ")
        length = 112

        with open(self.cfp_path + name1, 'rb') as f:
            img1 = Image.open(f).convert('RGB')
            img1 = img1.resize((length, length))


        with open(self.cfp_path + name2, 'rb') as f:
            img2 = Image.open(f).convert('RGB')
            img2 = img2.resize((length, length))

        try:
            with open(self.cfp_occ_path + name3, 'rb') as f:
                img2_occ = Image.open(f).convert('RGB')

        except:
            with open(self.cfp_path + name3, 'rb') as f:
                img2_occ = Image.open(f).convert('RGB')

        img2_occ = img2_occ.resize((length, length))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

            img2_occ = self.transform(img2_occ)

        return img1, img2, img2_occ, sameflag

    def __len__(self):
        return len(self.pairs)

class CFP_mask(Dataset):
    def __init__(self, config, transform=None):
        super(CFP_mask, self).__init__()
        self.cfp_path = config.DATASET.CFP_PATH
        self.cfp_occ_path = config.DATASET.CFP_OCC_PATH
        self.num_class = config.DATASET.CFP_CLASS
        self.mode = config.TEST.MODE
        self.pairs = self.get_pairs_lines(config.DATASET.CFP_PAIRS)

        self.transform = transform
        self.valid_check()
        self.num_pairs = len(self.pairs)

    def get_pairs_lines(self, path):
        with open(path) as f:
            pairs_lines = f.readlines()[:]
        return pairs_lines

    def valid_check(self):
        valid_pairs = []
        for pair in self.pairs:
            p = pair.replace('\n', '').split(',')

            if 2 == len(p):

                name1 = '{:03}'.format((int(p[0]) - 1) // 10 + 1) + '/' + '{:02}.jpg'.format((int(p[0]) - 1) % 10 + 1)
                name2 = '{:03}'.format((int(p[1]) - 1) // 10 + 1) + '/' + '{:02}.jpg'.format((int(p[1]) - 1) % 10 + 1)

                if abs((int(p[0]) - 1) // 10 - (int(p[1]) - 1) // 10) == 0:
                    sameflag = 1
                else:
                    sameflag = 0
            else:
                raise ValueError("WRONG LINE IN 'pairs.txt! ")


            if os.path.exists(self.cfp_path + name1)&os.path.exists(self.cfp_path + name2):
                valid_pairs.append(p)

        print('valid rate: {:.4f}'.format(len(valid_pairs) / len(self.pairs)))

        self.pairs = valid_pairs

    def __getitem__(self, index):
        p = self.pairs[index]

        if 2 == len(p):

            name1 = '{:03}'.format((int(p[0]) - 1) // 10 + 1) + '/' + '{:02}.jpg'.format((int(p[0]) - 1) % 10 + 1)
            name2 = '{:03}'.format((int(p[1]) - 1) // 10 + 1) + '/' + '{:02}.jpg'.format((int(p[1]) - 1) % 10 + 1)
            name3 = name2

            if abs((int(p[0]) - 1) // 10 - (int(p[1]) - 1) // 10) == 0:
                sameflag = 1
            else:
                sameflag = 0
        else:
            raise ValueError("WRONG LINE IN 'pairs.txt! ")
        length = 112
        try:
            with open(self.cfp_occ_path + name1, 'rb') as f:
                img1 = Image.open(f).convert('RGB')
        except:
            with open(self.cfp_path + name1, 'rb') as f:
                img1 = Image.open(f).convert('RGB')
        img1 = img1.resize((length, length))

        with open(self.cfp_path + name2, 'rb') as f:
            img2 = Image.open(f).convert('RGB')
            img2 = img2.resize((length, length))
        try:
            with open(self.cfp_occ_path + name3, 'rb') as f:
                img2_occ = Image.open(f).convert('RGB')
        except:
            with open(self.cfp_path + name3, 'rb') as f:
                img2_occ = Image.open(f).convert('RGB')
        img2_occ = img2_occ.resize((length, length))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img2_occ = self.transform(img2_occ)
        return img1, img2_occ, img2_occ, sameflag

    def __len__(self):
        return len(self.pairs)



class LFW_Image(Dataset):
    def __init__(self, config, transform=None):
        super(LFW_Image, self).__init__()
        self.lfw_path = config.DATASET.LFW_PATH
        self.lfw_occ_path = config.DATASET.LFW_OCC_PATH
        self.num_class = config.DATASET.LFW_CLASS
        self.mode = config.TEST.MODE 
        self.pairs = self.get_pairs_lines(config.DATASET.LFW_PAIRS)

        self.transform = transform
        self.valid_check()
        self.num_pairs = len(self.pairs)

    def get_pairs_lines(self, path):
        with open(path) as f:
            pairs_lines = f.readlines()[1:]
        return pairs_lines

    def valid_check(self):
        valid_pairs = []
        for pair in self.pairs:
            p = pair.replace('\n', '').split('\t')

            if 3 == len(p):
                sameflag = 1
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))#_surgical
            elif 4 == len(p):
                sameflag = 0
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))#_surgical
            else:
                raise ValueError("WRONG LINE IN 'pairs.txt! ")



            if os.path.exists(self.lfw_path + name2):
                valid_pairs.append(p)

        print('valid rate: {:.4f}'.format(len(valid_pairs) / len(self.pairs)))

        self.pairs = valid_pairs

    def __getitem__(self, index):
        # p = self.pairs[index].replace('\n', '').split('\t')
        p = self.pairs[index]

        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
            name3 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))#_surgical
        elif 4 == len(p):
            sameflag = 0
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
            name3 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))#_surgical
        else:
            raise ValueError("WRONG LINE IN 'pairs.txt! ")
        length = 112
        with open(self.lfw_path + name1, 'rb') as f:
            img1 =  Image.open(f).convert('RGB')
            img1 = img1.resize((length, length))
        with open(self.lfw_path + name2, 'rb') as f:
            img2 = Image.open(f).convert('RGB')
            img2 = img2.resize((length, length))
        try:
            with open(self.lfw_occ_path + name3, 'rb') as f:
                img2_occ =  Image.open(f).convert('RGB')
        except:
            with open(self.lfw_path + name3, 'rb') as f:
                img2_occ =  Image.open(f).convert('RGB')
        img2_occ = img2_occ.resize((length, length))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img2_occ = self.transform(img2_occ)
        return img1, img2, img2_occ, sameflag

    def __len__(self):
        return len(self.pairs)

class LFW_Mask_Image(Dataset):
    def __init__(self, config, transform=None):
        super(LFW_Mask_Image, self).__init__()
        self.lfw_path = config.DATASET.LFW_PATH
        self.lfw_occ_path = config.DATASET.LFW_OCC_PATH
        self.num_class = config.DATASET.LFW_CLASS
        self.mode = config.TEST.MODE
        self.pairs = self.get_pairs_lines(config.DATASET.LFW_PAIRS)

        self.transform = transform
        self.valid_check()
        self.num_pairs = len(self.pairs)

    def get_pairs_lines(self, path):
        with open(path) as f:
            pairs_lines = f.readlines()[1:]
        return pairs_lines

    def valid_check(self):
        valid_pairs = []
        for pair in self.pairs:
            p = pair.replace('\n', '').split('\t')

            if 3 == len(p):
                sameflag = 1
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))#_surgical
            elif 4 == len(p):
                sameflag = 0
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))#_surgical
            else:
                raise ValueError("WRONG LINE IN 'pairs.txt! ")



            if os.path.exists(self.lfw_occ_path + name2):
                valid_pairs.append(p)

        print('valid rate: {:.4f}'.format(len(valid_pairs) / len(self.pairs)))

        self.pairs = valid_pairs

    def __getitem__(self, index):
        p = self.pairs[index]

        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
            name3 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))#_surgical
        elif 4 == len(p):
            sameflag = 0
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
            name3 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))#_surgical
        else:
            raise ValueError("WRONG LINE IN 'pairs.txt! ")
        length = 112
        if os.path.isfile(self.lfw_occ_path + name1):
            image = Image.open(self.lfw_occ_path + name1).convert('RGB')
        else:
            image = Image.open(self.lfw_path + name1).convert('RGB')


        img1 = image.resize((length, length))
        image = Image.open(self.lfw_path + name2).convert('RGB')
        img2 = image.resize((length, length))
        image = Image.open(self.lfw_occ_path + name3).convert('RGB')
        img2_occ = image.resize((length, length))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img2_occ = self.transform(img2_occ)

        return img1, img2_occ, img2_occ, sameflag

    def __len__(self):
        return len(self.pairs)


class Masked_whn(Dataset):
    def __init__(self, config, transform=None):
        super(Masked_whn, self).__init__()
        self.mfr2_path = config.DATASET.whn_PATH
        self.mode = config.TEST.MODE
        self.pairs = self.get_pairs_lines(config.DATASET.whn_PAIRS)

        self.transform = transform
        self.valid_check()
        self.num_pairs = len(self.pairs)

    def get_pairs_lines(self, path):
        with open(path) as f:
            pairs_lines = f.readlines()[:]
        return pairs_lines

    def valid_check(self):
        valid_pairs = []
        for pair in self.pairs:
            p = pair.strip().split(' ')

            if 3 == len(p):
                sameflag = p[2]
                name1 = p[0]
                name2 = p[1]
            else:
                raise ValueError("WRONG LINE IN 'pairs.txt! ")
            if os.path.exists(self.mfr2_path + name1) and os.path.exists(self.mfr2_path + name2):
                valid_pairs.append(p)
            else:
                print(p)

        print('valid rate: {:.4f}'.format(len(valid_pairs) / len(self.pairs)))

        self.pairs = valid_pairs

    def __getitem__(self, index):
        p = self.pairs[index]

        if 3 == len(p):
            sameflag = int(p[2])
            name1 = p[0]
            name2 = p[1]
        else:
            raise ValueError("WRONG LINE IN 'pairs.txt! ")

        length = 112
        with open(self.mfr2_path + name1, 'rb') as f:
            img1 = Image.open(f).convert('RGB')
            img1 = img1.resize((length, length))

        with open(self.mfr2_path + name2, 'rb') as f:
            img2 =  Image.open(f).convert('RGB')
            img2 = img2.resize((length, length))

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img2_occ = img2
        return img1, img2, img2_occ, sameflag

    def __len__(self):
        return len(self.pairs)

class Age_db(Dataset):
    def __init__(self, config, transform=None):
        super(Age_db, self).__init__()
        self.Age_db_path = config.DATASET.agedb_PATH
        self.mode = config.TEST.MODE
        self.pairs = self.get_pairs_lines(config.DATASET.agedb_PAIRS)

        self.transform = transform
        self.valid_check()
        self.num_pairs = len(self.pairs)

    def get_pairs_lines(self, path):
        with open(path) as f:
            pairs_lines = f.readlines()[:]
        return pairs_lines

    def valid_check(self):
        valid_pairs = []
        for pair in self.pairs:
            p = pair.strip().split(' ')


            if 3 == len(p):
                sameflag = p[2]
                name1 = p[0]+'.jpg'
                name2 = p[1]+'.jpg'

            else:
                raise ValueError("WRONG LINE IN 'pairs.txt! ")


            if os.path.exists(self.Age_db_path + name1) and os.path.exists(self.Age_db_path + name2):
                valid_pairs.append(p)


        print('valid rate: {:.4f}'.format(len(valid_pairs) / len(self.pairs)))

        self.pairs = valid_pairs

    def __getitem__(self, index):
        p = self.pairs[index]

        if 3 == len(p):
            sameflag = int(p[2])
            name1 = p[0]+'.jpg'
            name2 = p[1]+'.jpg'

        else:
            raise ValueError("WRONG LINE IN 'pairs.txt! ")


        length = 112
        with open(self.Age_db_path + name1, 'rb') as f:
            img1 = Image.open(f).convert('RGB')
            img1 = img1.resize((length, length))

        with open(self.Age_db_path + name2, 'rb') as f:
            img2 =  Image.open(f).convert('RGB')
            img2 = img2.resize((length, length))


        if self.transform is not None:

            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img2_occ = img2
        return img1, img2, img2_occ, sameflag

    def __len__(self):
        return len(self.pairs)


class MFR2_Image(Dataset):
    def __init__(self, config, transform=None):
        super(MFR2_Image, self).__init__()
        self.mfr2_path = config.DATASET.MFR2_PATH
        self.mode = config.TEST.MODE 
        self.pairs = self.get_pairs_lines(config.DATASET.MFR2_PAIRS)

        self.transform = transform
        self.valid_check()
        self.num_pairs = len(self.pairs)

    def get_pairs_lines(self, path):
        with open(path) as f:
            pairs_lines = f.readlines()[:]
        return pairs_lines

    def valid_check(self):
        valid_pairs = []
        for pair in self.pairs:
            p = pair.strip().split(' ')

            if 3 == len(p):
                sameflag = 1
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[1]))
                name2 = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[2]))
            elif 4 == len(p):
                sameflag = 0
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[1]))
                name2 = p[2] + '/' + p[2] + '_' + '{:04}.png'.format(int(p[3]))
            else:
                raise ValueError("WRONG LINE IN 'pairs.txt! ")


            if os.path.exists(self.mfr2_path + name1) and os.path.exists(self.mfr2_path + name2):
                valid_pairs.append(p)

        print('valid rate: {:.4f}'.format(len(valid_pairs) / len(self.pairs)))

        self.pairs = valid_pairs

    def __getitem__(self, index):
        p = self.pairs[index]

        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[2]))
        elif 4 == len(p):
            sameflag = 0
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.png'.format(int(p[3]))
        else:
            raise ValueError("WRONG LINE IN 'pairs.txt! ")

        length = 112
        with open(self.mfr2_path + name1, 'rb') as f:
            img1 =  Image.open(f).convert('RGB')
            img1 = img1.resize((length, length))

        with open(self.mfr2_path + name2, 'rb') as f:
            img2 =  Image.open(f).convert('RGB')
            img2 = img2.resize((length, length))


        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img2_occ = img2 
        return img1, img2, img2_occ, sameflag

    def __len__(self):
        return len(self.pairs)



