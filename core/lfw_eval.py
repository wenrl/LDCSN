from PIL import Image
import numpy as np
import copy
import os

from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import logging
from tensorboardX import SummaryWriter

# cudnn.benchmark = True

logger = logging.getLogger(__name__)

def extractDeepFeature(img, model, is_gray, mask=None, binary=False):
    img = img.to('cuda')
    # img = img.to('cpu')
    fc = model(img)
    fc = fc[0].to('cpu').squeeze()
    # fc = fc.to('cpu').squeeze()
    return fc


def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[0]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[1]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    # print(y_true,y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

def cosine_similarity(f1, f2):
    # compute cosine_similarity for 2-D array
    f1 = f1.numpy()
    f2 = f2.numpy()

    A = np.sum(f1*f2, axis=1)
    B = np.linalg.norm(f1, axis=1) * np.linalg.norm(f2, axis=1) + 1e-5

    return A / B
#
# def binary_mask(mask, thres=0.3):
#     ones = torch.ones_like(mask)
#     zeros = torch.zeros_like(mask)
#     mask = torch.where(mask > thres, ones, zeros)
#     return mask

# def soft_binary_mask(mask, thres=0.3):
#     zeros = torch.zeros_like(mask)
#     mask = torch.where(mask > thres, mask, zeros)
#     # expand to [0, 1]
#     # mask = (mask - torch.min(mask)) / (torch.max(mask) - torch.min(mask))
#     return mask

# def mask_ratio(mask, ratio=0.2):
#     index = int(ratio * mask.nelement())
#     thres = torch.sort(mask.flatten())[0][index].item()
#     return binary_mask(mask, thres)

# def cal_mask_stat(mask, num=11):
#     grid = np.linspace(0, 1, num)
#     mask_stats = np.zeros((num,))
#
#     for i in range(num-1):
#         low = grid[i]
#         high = grid[i+1]
#         A = mask >= low
#         B = mask < high
#         count = np.sum(A * B)
#         mask_stats[i] = count
#     mask_stats[num-1] = mask.size - np.sum(mask_stats)
#     return mask_stats

def compute_distance(img1, img2, model, flag, is_gray):
        # f2 = extractDeepFeature(img2, model, is_gray, None)
        # f1 = extractDeepFeature(img1, model, is_gray, None)

        # mask_occ = mask2.cpu().detach().numpy()
        # mask_clean = mask1.cpu().detach().numpy()
        #
        # if soft_binary:
        #     mask1 = soft_binary_mask(mask1, binary_thres)
        #     mask2 = soft_binary_mask(mask2, binary_thres)
        # else:
        #     mask1 = binary_mask(mask1, binary_thres)
        #     mask2 = binary_mask(mask2, binary_thres)

        f2 = extractDeepFeature(img2, model, is_gray)
        f1 = extractDeepFeature(img1, model, is_gray)

        # if mode != 'Mask':
        #     f1_mask, f2_mask = f1, f2

        # if indicator == 'Baseline':
        distance = cosine_similarity(f1, f2)
        # else:
            # distance = cosine_similarity(f1_mask, f2_mask)

        flag = flag.squeeze().numpy()
        # print(distance.shape, flag.shape)
        return np.stack((distance, flag), axis=1)

def tar_far(far, predicts, name):
    posi_scores = []
    nega_scores = []
    for pairs in predicts:
        if pairs[1] == 0:
            nega_scores.append(pairs[0])
        else:
            posi_scores.append(pairs[0])
    posi_scores.sort(reverse=True)
    nega_scores.sort(reverse=True)

    thres_idx = int(len(nega_scores) * far)
    threshold = nega_scores[thres_idx]

    num_correct = np.array(np.array(posi_scores) >= threshold, dtype=int).sum()
    tar = float(num_correct) / len(posi_scores)
    logger.info('{}: far {} leads to tar {:.4f}'.format(name, far, tar))
    return tar

def obtain_acc(predicts, num_class, name, start):
    accuracy = []
    thd = []
    folds = KFold(n=num_class, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    end = time.time()
    time_used = (end - start) / 60.0
    logger.info('{}_LFW_ACC={:.4f} std={:.4f} thd={:.4f} time_used={:.4f} mins'.format(name, np.mean(accuracy), np.std(accuracy), np.mean(thd), time_used))
    return np.mean(accuracy)


def eval(model, model_path, config, test_loader, tb_log_dir, epoch, is_gray=False):
    indicator = 'LDCSN'
    if model_path:
        # if 'baseline' in model_path or '_occ_' in model_path:
        #     indicator = 'Baseline'
        logger.info('Testing model:{}, model_path:{}'.format(model_path, indicator))
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']

        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=True)

    predicts = np.zeros(shape=(len(test_loader.dataset), 2))
    predicts_mask = np.zeros(shape=(len(test_loader.dataset), 2))

    model.eval()
    start = time.time()

    cur = 0
    with torch.no_grad():
        for batch_idx, (img1, img2, img2_occ, flag) in enumerate(test_loader):
            predicts[cur:cur+flag.shape[0]] = compute_distance(img1, img2, model, flag, is_gray)
            predicts_mask[cur:cur+flag.shape[0]] = compute_distance(img1, img2_occ, model, flag, is_gray)
            cur += flag.shape[0]
    assert cur == predicts.shape[0]
    # print('predicts,','\n',predicts[:10])
    # print('num_pairs,', '\n', test_loader.dataset.num_pairs)
    accuracy = obtain_acc(predicts, test_loader.dataset.num_pairs, 'Clean', start)
    accuracy_mask = obtain_acc(predicts_mask, test_loader.dataset.num_pairs, 'Occ', start)


    # visualize the masks stats
    writer = SummaryWriter(tb_log_dir)
    writer.add_scalar('Clean_LFW_Acc', np.mean(accuracy), epoch)
    writer.add_scalar('Mask_LFW_Acc', np.mean(accuracy_mask), epoch)
    writer.close()


    return np.mean(accuracy), np.mean(accuracy_mask)

