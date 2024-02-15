import argparse
import json
import os
import numpy as np

import torch
import torch.utils.data
import torch.optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import core.utils as utils
from core.config import config
from core.config import update_config
from core.lfw_eval import eval as lfw_eval
from datasets.dataset import LFW_Image, LFW_Mask_Image, MFR2_Image, CFP, CFP_mask, Age_db, Masked_whn, RWMFD



from MMHSA.fusionnet_reduction import LResNet50_LDCSN as LDCSN_model
# from MMHSA.fusionnet_reduction_conv import LResNet50_LDCSN as LDCSN_model




# setup random seed
torch.manual_seed(0)
np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch End2End Occluded Face')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--batch_size', help='batch size', type=int)
    parser.add_argument('--pretrained', help='whether use pretrained model', type=str)
    parser.add_argument('--debug', help='whether debug', default=0, type=int)
    parser.add_argument('--model', help=' model name', type=str)
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.TRAIN.GPUS = args.gpus
    if args.workers:
        config.TRAIN.WORKERS = args.workers
    if args.model:
        print('update model type')
        config.TRAIN.MODEL = args.model
    if args.batch_size:
        print('update batch_size')
        config.TRAIN.BATCH_SIZE = args.batch_size
        config.TEST.BATCH_SIZE = args.batch_size
    if args.pretrained == 'No':
        print('update pretrained')
        config.NETWORK.PRETRAINED = ''


def main():
    # --------------------------------------model----------------------------------------
    args = parse_args()
    reset_config(config, args)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.TRAIN.GPUS
    gpus = [int(i) for i in config.TRAIN.GPUS.split(',')]
    gpus = range(len(gpus))

    logger, final_output_dir, tb_log_dir = utils.create_temp_logger()

    logger.info('Configs: \n' + json.dumps(config, indent=4, sort_keys=True))
    logger.info('Args: \n' + json.dumps(vars(args), indent=4, sort_keys=True))

    test_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])



    def accu_dataset(dataset):
        test_loader = torch.utils.data.DataLoader(
            dataset(config, test_transform),
            batch_size=config.TEST.BATCH_SIZE * len(gpus),
            shuffle=config.TEST.SHUFFLE,
            num_workers=config.TEST.WORKERS,
            pin_memory=True)
        return test_loader

    model_root = 'C:/Users/86137/Desktop/experiments/wrl/pretrained'

    model_list = [
        # fusion net
        'model_best_fusionnet.pth.tar'
    ]

    for model_name in model_list:
        model = LDCSN_model()
        # model = LDCSN_model().eval()
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

        model_path = os.path.join(model_root, model_name)
        #single dataset testing
        # lfw_eval(model, model_path, config, test_loader, 'temp', 0)
        # all testing
        dataset_list = [LFW_Image, LFW_Mask_Image, MFR2_Image, CFP, CFP_mask, Masked_whn]
        for dataset in dataset_list:
            test_loader = accu_dataset(dataset)
            lfw_eval(model, model_path, config, test_loader, 'temp', 0)
        print(' ')


if __name__ == '__main__':
    main()
