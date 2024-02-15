import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import json
import shutil
import logging
import argparse
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
from core.functions import train, train_KD, train_ufa
from core.lfw_eval import eval as lfw_eval
from Training_dataset.datasets import Dataset, Dataset_align, Dataset_align_plus, Dataset_VGG_align, Dataset_align_KD
from datasets.dataset import LFW_Image,MFR2_Image,LFW_Mask_Image
from models.metrics import ArcMarginProduct, ModifiedGDC, CosMarginProduct, ElasticArcFace, SphereFace2, SphereMarginProduct
#LDCSN
from MMHSA.fusionnet_reduction import LResNet50_LDCSN as LDCSN_model
# from EMHSA_Conv.EMHSA_DWConv_UFA_ConvSAs.fusionnet.fusionnet_reduction_conv import LResNet50E_IR as LResNet50E_IR_FPN

# setup random seed
torch.manual_seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch MaskFace')
    parser.add_argument('--cfg', help='experiment configure file name', required=False, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    parser.add_argument('--frequent', help='frequency of logging', default=config.TRAIN.PRINT_FREQ, type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--batch_size', help='batch size', type=int)
    parser.add_argument('--lr', help='init learning rate', type=float)
    parser.add_argument('--optim', help='optimizer type', type=str)
    parser.add_argument('--pretrained', help='whether use pretrained model', type=str)
    parser.add_argument('--debug', help='whether debug', default=0, type=int)
    parser.add_argument('--model', help=' model name', type=str)
    parser.add_argument('--loss', help=' loss type', type=str)
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
    if args.loss:
        print('update loss type')
        config.LOSS.TYPE = args.loss
    if args.batch_size:
        print('update batch_size')
        config.TRAIN.BATCH_SIZE = args.batch_size
        config.TEST.BATCH_SIZE = args.batch_size
    if args.lr:
        print('update learning rate')
        config.TRAIN.LR = args.lr
    if args.pretrained =='False':
        print('update pretrained')
        config.NETWORK.PRETRAINED = ''
    if args.optim:
        print('update optimizer type')
        config.TRAIN.OPTIMIZER = args.optim
        if args.optim == 'adam':
            config.TRAIN.LR = 1e-4

def main():
    # --------------------------------------model----------------------------------------
    args = parse_args()
    reset_config(config, args)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.TRAIN.GPUS

    if args.debug:
        logger, final_output_dir, tb_log_dir = utils.create_temp_logger()
    else:
        logger, final_output_dir, tb_log_dir = utils.create_logger(
            config, args.cfg, 'train')

    model = {
        'LDCSN_model': LDCSN_model()
    }[config.TRAIN.MODEL]

    # choose the type of loss 512 is dimension of feature
    classifier = {
        'ArcMargin': ArcMarginProduct(512, config.DATASET.NUM_CLASS),
        'CosMargin': CosMarginProduct(512, config.DATASET.NUM_CLASS),
        'SphereFace2': SphereFace2(512, config.DATASET.NUM_CLASS),
        'SphereMargin': SphereMarginProduct(512, config.DATASET.NUM_CLASS),
        'ElasticMargin':ElasticArcFace(512, config.DATASET.NUM_CLASS),
    }[config.LOSS.TYPE]
    #ufa
    classifier_ufa = CosMarginProduct(512, config.DATASET.NUM_CLASS)

    # --------------------------------loss function and optimizer-----------------------------
    # optimizer_sgd = torch.optim.SGD([{'params': model.parameters()}],#, {'params': classifier.parameters()}
    optimizer_sgd = torch.optim.SGD([{'params': model.parameters()}, {'params': classifier.parameters()}],#
                                                                  lr=config.TRAIN.LR,
                                                                  momentum=config.TRAIN.MOMENTUM,
                                                                  weight_decay=config.TRAIN.WD)
    optimizer_adam = torch.optim.Adam([{'params': model.parameters()}, {'params': classifier.parameters()}],
                                        lr=config.TRAIN.LR)

    optimizer_sgd_ufa = torch.optim.SGD([{'params': classifier_ufa.parameters()}],
                                        lr=config.TRAIN.LR,
                                        momentum=config.TRAIN.MOMENTUM,
                                        weight_decay=config.TRAIN.WD)

    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optimizer_sgd
        # ufa
        optimizer_ufa = optimizer_sgd_ufa
    elif config.TRAIN.OPTIMIZER == 'adam':
        optimizer = optimizer_adam
        # ufa
        optimizer_ufa = optimizer_sgd_ufa
    else:
        raise ValueError('unknown optimizer type')

    criterion = torch.nn.CrossEntropyLoss().cuda()
    # ufa
    criterion_ufa = torch.nn.CrossEntropyLoss().cuda()
    # criterion = torch.nn.CrossEntropyLoss().cpu()
    start_epoch = config.TRAIN.START_EPOCH
    #骨干网络预训练
    if config.NETWORK.PRETRAINED:
        model, classifier = utils.load_pretrained2(model, classifier, final_output_dir)
        # ufa classifier_ufa pretraining
        # model, classifier, _ = utils.load_pretrained1(model, classifier, classifier_ufa)

    if config.TRAIN.RESUME:
        start_epoch, model, optimizer, classifier = \
            utils.load_checkpoint(model, optimizer, classifier, final_output_dir)
    # if config.TRAIN.RESUME:
    #     model, classifier = utils.load_pretrained1(model, optimizer, classifier, final_output_dir)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)

    # ufa
    lr_scheduler_ufa = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_ufa, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    # ufa
    for state in optimizer_ufa.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    gpus = [int(i) for i in config.TRAIN.GPUS.split(',')]
    gpus = range(len(gpus))

    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    cudnn.benchmark = True
    # model = model.cpu()
    classifier = torch.nn.DataParallel(classifier, device_ids=gpus).cuda()
    #ufa
    classifier_ufa = torch.nn.DataParallel(classifier_ufa, device_ids=gpus).cuda()
    # classifier = classifier.cpu()

    # logger.info(model)
    logger.info('Configs: \n' + json.dumps(config, indent=4, sort_keys=True))
    logger.info('Args: \n' + json.dumps(vars(args), indent=4, sort_keys=True))

    # ------------------------------------load image---------------------------------------
    if config.TRAIN.MODE in ['Mask', 'Occ']:
        train_transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
            transforms.RandomCrop(config.NETWORK.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])


    dataset = Dataset_align(root='/home/root1/文档/CASIA/casia_align/',
                            data_list_file='/home/root1/文档/CASIA/casia_align_random.txt',
                            mask=True)

    train_loader = torch.utils.data.DataLoader(

        dataset=dataset,
        batch_size=config.TRAIN.BATCH_SIZE*len(gpus), 
        shuffle=config.TRAIN.SHUFFLE,#True
        num_workers=config.TRAIN.WORKERS, #0
        pin_memory=True,
        # drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        LFW_Image(config, test_transform),
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=config.TEST.SHUFFLE,#,
        num_workers= config.TEST.WORKERS,
        pin_memory=True)

    logger.info('length of train Database: ' + str(len(train_loader.dataset)) + '  Batches: ' + str(len(train_loader)))
    logger.info('Number of Identities: ' + str(config.DATASET.NUM_CLASS))

    # ----------------------------------------train----------------------------------------
    start = time.time()
    best_acc = 0.0
    best_model = False

    w=300
    # best_keep = [0, 0, 0]
    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):

        # train_KD(train_loader, model, occmodel, classifier, criterion, optimizer,occoptimizer, epoch, tb_log_dir, config,w)
        # train_ufa(train_loader, model, classifier, criterion, criterion_ufa, optimizer, epoch, tb_log_dir, config, classifier_ufa, optimizer_ufa, w)#, classifier_ufa  , optimizer_ufa
        train(train_loader, model, classifier, criterion, optimizer, epoch, tb_log_dir,config)


        acc, acc_occ= lfw_eval(model, None, config, test_loader, tb_log_dir, epoch)

        perf_acc = acc if config.TRAIN.MODE == 'Mask' else acc_occ

        lr_scheduler.step()
        # ufa
        # lr_scheduler_ufa.step()

        if perf_acc > best_acc:
            best_acc = perf_acc
            best_keep = [acc, acc_occ]
            # filename = 'checkpoint.pth.tar'
            best_model = True

        else:
            best_model = False
            # best_keep = [acc, acc_occ]
        # if epoch >= 10:
        #     best_model = True
        #     filename = 'checkpoint{}.pth.tar'.format(epoch)

        logger.info('current best accuracy {:.5f}'.format(best_acc))
        logger.info('saving checkpoint to {}'.format(final_output_dir))
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'model': args.cfg,
            'state_dict': model.module.state_dict(),
            'perf': perf_acc,
            'optimizer': optimizer.state_dict(),
            'classifier': classifier.module.state_dict(),
        }, best_model, final_output_dir)#, filename=filename
        # ufa
        # utils.save_checkpoint({
        #     'epoch': epoch + 1,
        #     'model': args.cfg,
        #     'state_dict': model.module.state_dict(),
        #     'perf': perf_acc,
        #     'optimizer': optimizer.state_dict(),
        #     'classifier': classifier.module.state_dict(),
        #     'classifier_ufa': classifier_ufa.module.state_dict(),
        # }, best_model, final_output_dir)

    # save best model with its acc
    shutil.move(os.path.join(final_output_dir, 'model_best.pth.tar'),
                os.path.join(final_output_dir, 'model_best_{:.4f}_{:.4f}.pth.tar'.format(best_keep[0], best_keep[1])))

    end = time.time()
    time_used = (end - start) / 3600.0
    logger.info('Done Training, Consumed {:.2f} hours'.format(time_used))

if __name__ == '__main__':
    main()
