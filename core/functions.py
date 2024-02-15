import os
import time
import logging
import numpy as np
from copy import copy
import cv2
import torch
from torchviz import make_dot
from torchsummary import summary
import torch.nn.functional as F
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
criterien_KL = torch.nn.KLDivLoss(reduction='batchmean').cuda()
criterien_pene = torch.nn.MSELoss().cuda()


def train(train_loader, model, classifier, criterion, optimizer, epoch, tb_log_dir, config):
    # model = model.cpu()

    model.train()

    time_curr = time.time()
    loss_display = 0.0
    loss_cls_dis = 0.0


    for batch_idx, data in enumerate(train_loader):
        img, label = data
        img, label = img.cuda(), label.cuda()
        # img, label = img.cpu(), label.cpu()
        features = model(img)
        feature = features
        # cosface
        output = classifier(feature, label)
        loss = criterion(output, label)

        loss_display += loss.item()
        loss_cls_dis = 0

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        iters = epoch * len(train_loader) + batch_idx

        if iters % config.TRAIN.PRINT_FREQ == 0 and iters != 0:
            
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(float))

            time_used = time.time() - time_curr
            if batch_idx < config.TRAIN.PRINT_FREQ:
                num_freq = batch_idx + 1
            else:
                num_freq = config.TRAIN.PRINT_FREQ
            speed = num_freq / time_used
            loss_display /= num_freq
            loss_cls_dis /= num_freq

            INFO = ' Margin: {:.2f}, Scale: {:.2f}'.format(classifier.module.m, classifier.module.s)
            logger.info(
                'Train Epoch: {} [{:03}/{} ({:.0f}%)]{:05}, Loss: {:.6f}, Acc: {:.4f}, Elapsed time: {:.4f}s, Batches/s {:.4f}'.format(
                    epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader),
                    iters, loss_display, acc, time_used, speed) + INFO)
            with SummaryWriter(tb_log_dir) as sw:
                sw.add_scalar('TRAIN_LOSS', loss_display, iters)
                sw.add_scalar('TRAIN_ACC', acc, iters)
            time_curr = time.time()
            loss_display = 0.0
            loss_cls_dis = 0.0

def train_ufa(train_loader, model, classifier, criterion, criterion_ufa, optimizer, epoch, tb_log_dir, config, classifier_ufa, optimizer_ufa, w):#
    # model = model.cpu()

    model.train()

    time_curr = time.time()
    loss_display = 0.0
    loss_cls_dis = 0.0
    loss_kl_dis = 0.0

    for batch_idx, data in enumerate(train_loader):

        img, label = data
        img, label = img.cuda(), label.cuda()
        features = model(img)

        # compute output

        feature = features[0]
        ufa = features[1]
        output = classifier(feature, label)
        loss_cls = criterion(output, label)
            # ufa
        output_ufa = classifier_ufa(ufa, label)
        loss_kl = criterien_KL(F.log_softmax(output_ufa), F.softmax(output))
        loss = loss_cls+loss_kl


        loss_display += loss.item()

        loss_cls_dis += loss_cls.item()
        loss_kl_dis += loss_kl.item()
        # compute gradient and do SGD step

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        optimizer_ufa.step()
        optimizer_ufa.zero_grad()
        # print('iters')
        iters = epoch * len(train_loader) + batch_idx

        if iters % config.TRAIN.PRINT_FREQ == 0 and iters != 0:
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(float))

            time_used = time.time() - time_curr
            if batch_idx < config.TRAIN.PRINT_FREQ:
                num_freq = batch_idx + 1
            else:
                num_freq = config.TRAIN.PRINT_FREQ
            speed = num_freq / time_used
            loss_display /= num_freq
            loss_cls_dis /= num_freq
            loss_pred_dis /= num_freq

            INFO = ' Margin: {:.2f}, Scale: {:.2f}'.format(classifier.module.m, classifier.module.s)
            logger.info(
                'Train Epoch: {} [{:03}/{} ({:.0f}%)]{:05}, Loss: {:.6f}, Acc: {:.4f}, Elapsed time: {:.4f}s, Batches/s {:.4f}'.format(
                    epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader),
                    iters, loss_display, acc, time_used, speed) + INFO)
            if config.TRAIN.MODE == 'Clean':
                print('LDCSN')
                logger.info('Cls Loss: {:.4f}; Pred Loss: {:.4f}*{}'.format(loss_cls_dis, loss_pred_dis, config.LOSS.WEIGHT_PRED))
            with SummaryWriter(tb_log_dir) as sw:
                sw.add_scalar('TRAIN_LOSS', loss_display, iters)
                sw.add_scalar('TRAIN_ACC', acc, iters)

            time_curr = time.time()
            loss_display = 0.0
            loss_cls_dis = 0.0
            loss_pred_dis = 0.0


def occ_train(features, label, config, classifier, criterion):
    fc_mask, mask, vec, fc = features

    output = classifier(fc_mask, label)
    loss_cls = criterion(output, label)

    # loss_pred = criterion(vec, mask_label)
    preds = vec.cpu().detach().numpy()
    preds = np.argmax(preds, axis=1)

    loss = loss_cls

    return output, loss, loss_cls, mask, preds
classifier1 = torch.nn.MSELoss()
criterion1 = torch.nn.MSELoss().cuda()
# criterion2 = torch.nn.CosineEmbeddingLoss().cuda()
# criterion3 = torch.nn.SmoothL1Loss().cuda()
# torch.nn.HingeEmbeddingLoss().cuda()
def train_KD(train_loader, model, occmodel, classifier, criterion, optimizer, occoptimizer, epoch, tb_log_dir, config,w):
    # model = model.cpu()
    # summary(model, torch.tensor((3, 112, 96)).cuda())
    occmodel.eval()
    model.train()
    classifier.train()
    time_curr = time.time()
    loss_display = 0.0
    loss_cls_dis = 0.0
    loss_pred_dis = 0.0
    # a = torch.tensor([1.0]).cuda()
    # print(len(train_loader))
    for batch_idx, data in enumerate(train_loader):
        img,occimg, label = data
        img,occimg, label = img.cuda(),occimg.cuda(), label.cuda()
        # mask_label = mask_label.cuda()
        # print(img.shape)
        # occmodel = copy(model)
        # img, label = img.cpu(), label.cpu()
        # mask_label = mask_label.cpu()
        with torch.no_grad():
            occfeatures = F.normalize(occmodel(img))
            # occfeatures = occmodel(img)
        features = F.normalize(model(occimg))
        # features = model(occimg)
        # compute output
        if config.TRAIN.MODE == 'Clean' or config.TRAIN.MODE == 'Occ':
            feature = features
            occfeature = occfeatures
            output = classifier(feature, label)
            loss1 = criterion(output, label)
            loss_pred = w*criterion1(feature, occfeature)#+criterion2(feature, occfeature,a)
            # loss2 = (a.cuda()-criterion2(feature, occfeature))
            loss = loss1 + loss_pred# + loss2

        elif config.TRAIN.MODE == 'Mask':
            output, loss, loss_cls, mask, preds = occ_train(features, label, config, classifier, criterion)
        else:
            raise ValueError('Unknown training mode!')

        loss_display += loss.item()
        if config.TRAIN.MODE == 'Clean':
            loss_cls_dis += loss1.item()
            loss_pred_dis += loss_pred.item()
        else:
            loss_cls_dis = 0
            loss_pred_dis = 0
        # compute gradient and do SGD step

        loss.backward()
        optimizer.step()
        occoptimizer.step()
        optimizer.zero_grad()
        occoptimizer.zero_grad()
        # print('iters')
        iters = epoch * len(train_loader) + batch_idx

        if iters % config.TRAIN.PRINT_FREQ == 0 and iters != 0:

            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(float))

            time_used = time.time() - time_curr
            if batch_idx < config.TRAIN.PRINT_FREQ:
                num_freq = batch_idx + 1
            else:
                num_freq = config.TRAIN.PRINT_FREQ
            speed = num_freq / time_used
            loss_display /= num_freq
            loss_cls_dis /= num_freq
            loss_pred_dis /= num_freq

            INFO = ' Margin: {:.2f}, Scale: {:.2f}'.format(classifier.module.m, classifier.module.s)
            # INFO = ' Margin: {:.2f}, Scale: {:.2f}'.format(classifier.m, classifier.s)
            logger.info(
                'Train Epoch: {} [{:03}/{} ({:.0f}%)]{:05}, Loss: {:.6f}, Acc: {:.4f}, Elapsed time: {:.4f}s, Batches/s {:.4f}'.format(
                    epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader),
                    iters, loss_display, acc, time_used, speed) + INFO)
            if config.TRAIN.MODE == 'Clean':
                # print('Mask')
                logger.info('Cls Loss: {:.4f}; Pred Loss: {:.4f}*{}'.format(loss_cls_dis, loss_pred_dis, w))
            with SummaryWriter(tb_log_dir) as sw:
                sw.add_scalar('TRAIN_LOSS', loss_display, iters)
                sw.add_scalar('TRAIN_ACC', acc, iters)
                if config.TRAIN.MODE == 'Mask':
                    sw.add_scalar('CLS_LOSS', loss_cls_dis, iters)
                    # sw.add_scalar('PRED_LOSS', 0, iters)
            time_curr = time.time()
            loss_display = 0.0
            loss_cls_dis = 0.0
            loss_pred_dis = 0.0
