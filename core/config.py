from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict

pwd = './'
config= edict()

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# common params for NETWORK
config.NETWORK = edict()
config.NETWORK.PRETRAINED = ''
config.NETWORK.IMAGE_SIZE = (112, 96)
config.NETWORK.WEIGHT_MODEL = ''
config.NETWORK.WEIGHT_FC = ''
config.NETWORK.FACTOR = 1

config.LOSS = edict()
config.LOSS.TYPE = 'ArcMargin' # 'CosMargin'
config.LOSS.WEIGHT_DIFF = 10
config.LOSS.WEIGHT_MASK = 100
config.LOSS.WEIGHT_PRED = 10.0

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = pwd + 'datasets/CASIA-WebFace/'
config.DATASET.LMDB_FILE = r'C:\Users\86137\Desktop\task\lmdb\train_10572.lmdb'# 'lmdb/CASIA-112x96-LMDB1/train_10572.lmdb'
config.DATASET.TRAIN_DATASET = 'WebFace'
config.DATASET.TEST_DATASET = 'LFW'
config.DATASET.NUM_CLASS = 10572
config.DATASET.IS_GRAY = False

# LFW related
config.DATASET.LFW_PATH = 'C:/Users/86137/Desktop/task/data/datasets/lfw-occ/lfw-112X112/'#R'C:\Users\cw\Desktop\数据集\lfw_masked'
# config.DATASET.LFW_OCC_PATH = 'datasets/lfw/lfw-112X96_occ/'
# config.DATASET.LFW_OCC_PATH ='data/datasets/lfw-occ/lfw-112X96-new-occ/'
config.DATASET.LFW_OCC_PATH ='C:/Users/86137/Desktop/task/data/datasets/lfw-occ/lfw-112X112__masked/'
config.DATASET.LFW_PAIRS ='C:/Users/86137/Desktop/task/data/datasets/lfw-occ/pairs.txt'
config.DATASET.LFW_CLASS = 6000

#CFP related
config.DATASET.CFP_PATH = 'C:/Users/86137/Desktop/task/data/datasets/cfp-dataset/cfp-ff_align/'
config.DATASET.CFP_OCC_PATH = 'C:/Users/86137/Desktop/task/data/datasets/cfp-dataset/cfp-ff_align__masked/'
# config.DATASET.CFP_PAIRS = r'C:/Users/86137/Desktop/task/data/datasets/cfp-dataset/cfp-ff_pairs.txt'

# config.DATASET.CFP_PATH = 'C:/Users/86137/Desktop/task/data/datasets/cfp-dataset/new_data/'
# config.DATASET.CFP_OCC_PATH = 'C:/Users/86137/Desktop/task/data/datasets/cfp-dataset/new_data__masked/'
config.DATASET.CFP_PAIRS = r'C:/Users/86137/Desktop/task/data/datasets/cfp-dataset/cfp-ff_std.txt'
config.DATASET.CFP_CLASS = 7000

# RMFD related
config.DATASET.RMFD_PATH = 'data/datasets/RMFD/masked_whn_algin_112x96_clean/'
config.DATASET.RMFD_PAIRS ='data/datasets/RMFD/masked_pairs.txt'

# MFR2 related
config.DATASET.MFR2_PATH = r'C:/Users/86137/Desktop/task/data/datasets/mfr2/'
config.DATASET.MFR2_PAIRS = 'C:/Users/86137/Desktop/task/data/datasets/mfr2/pairs.txt'

# Masked_whn related
config.DATASET.whn_PATH = r'C:/Users/86137/Desktop/task/data/datasets/masked_whn/'#whn-datasets/
config.DATASET.whn_PAIRS = 'C:/Users/86137/Desktop/task/data/datasets/whn-datasets/masked_pairs.txt'

# agedb related
config.DATASET.agedb_PATH = r'D:\dataset\AgeDB\03_Protocol_Images/'#whn-datasets/
config.DATASET.agedb_PAIRS = r'C:\Users\86137\Desktop\MFR_EMHSA\wrl_code\mat2txt\output1.txt'

# O_LFW related
config.DATASET.OLFW_PATH = 'data/datasets/O_LFW/O_O_Align_112x96/'
config.DATASET.OLFW_PAIRS ='data/datasets/O_LFW/O_O_Align_112x96/O_O_Flag_LR.txt'

# training data augmentation
config.DATASET.SCALE_FACTOR = 0
config.DATASET.ROT_FACTOR = 20

# train
config.TRAIN = edict()
config.TRAIN.OUTPUT_DIR = 'output'
config.TRAIN.LOG_DIR = 'log'
config.TRAIN.BACKBONE_MODEL = 'LResNet50E_IR'
config.TRAIN.MODEL = 'LResNet50E_IR_Occ'
config.TRAIN.MODE = 'Clean'
config.TRAIN.GPUS = '0'
config.TRAIN.WORKERS = 0
config.TRAIN.PRINT_FREQ = 100
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [15, 30]
config.TRAIN.LR = 0.1
config.TRAIN.LR_FREEZE = 0.1

config.TRAIN.OPTIMIZER = 'sgd'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0005
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.START_EPOCH = 0
config.TRAIN.END_EPOCH = 40
config.TRAIN.RESUME = ''
config.TRAIN.RESUME_FC = ''
config.TRAIN.BATCH_SIZE = 256
config.TRAIN.SHUFFLE = True
config.TRAIN.IOU = [0.2, 0.8]
config.TRAIN.ABLATION = False
config.TRAIN.NUM_MASK = 101
config.TRAIN.PATTERN = 4
config.TRAIN.BINARY_THRES = 0
config.TRAIN.SOFT_BINARY = 0

# testing
config.TEST = edict()
config.TEST.BATCH_SIZE = 512
config.TEST.SHUFFLE = False
config.TEST.WORKERS = 0
config.TEST.STATE = ''
config.TEST.MODEL_FILE = ''
config.TEST.MODE = 'Clean'
# Occ means using normal to test occ imgs, and Mask means using mask trained model to test occ imgs with mask
#user
# config.cfg = 'experiments/CASIA-112x96-LMDB-Mask.yaml'
# config.model = 'LResNet50E_IR_FPN'
# config.batch_size=128
# config.gpus='0'
# config.lr = 0.01
# config.weight_pred = 1
# config.optim='sgd'
# config.ratio=3
# config.pattern =5
# config.debug=0
#user

def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file=''):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    config_file = config.DATASET.LMDB_FILE.split('/')[-2] + '.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.BBOX_FILE = os.path.join(config.DATA_DIR, config.TEST.BBOX_FILE)

    config.NETWORK.PRETRAINED = os.path.join(config.DATA_DIR,
                                             config.NETWORK.PRETRAINED)


def get_model_name(cfg):
    name = '{model}_{loss_type}'.format(
        model=cfg.TRAIN.MODEL, loss_type=cfg.LOSS.TYPE)

    full_name = '{dataset}_{name}'.format(
        dataset=os.path.basename(config.DATASET.ROOT),
        name=name)

    return name, full_name


if __name__ == '__main__':
    # import sys
    # gen_config(sys.argv[1])
    # print(config.TRAIN.WORKERS)
    gen_config()
