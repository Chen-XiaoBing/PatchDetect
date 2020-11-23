import os
import re
import pdb
import torch
import models
import logging

import numpy as np
import torch.nn.functional as F

from tqdm import tqdm, trange
from mylogger import get_mylogger
from collections import Counter


class Configuration():
    def __init__(self):
        self.ARCH = "ResNet18"
        self.PATCH_SIZE = 7
        self.FILENUM = 10000
        self.FILEDIR = './result/attack/image_specific/ResNet18/ResNet18_{}/'.format(
            self.PATCH_SIZE)
        self.MODEL_NAME = './result/models/{}_{}/model_best.pth.tar'.format(
            self.ARCH, self.PATCH_SIZE)
        self.TARGET_LABLE = 5 


def collect_images(filedir, filenum):
    files = os.listdir(filedir)
    benign_files = ['' for i in range(filenum)]
    advers_files = ['' for i in range(filenum)]
    for i in files:
        if not i.endswith('.jpg'):
            continue
        idx = int(i.split('_')[-2])
        if idx >= filenum:
            continue
        if i.startswith('ori_'):
            benign_files[idx] = i
        elif i.startswith('patched_'):
            advers_files[idx] = i

    fimages = {'benign': benign_files, 'patch': advers_files}
    # for i in range(filenum):
    #     logger.info(i, benign_files[i], advers_files[i])
    return fimages


def mdr_vote(prob_grid, class_grid, file_idx='', prefix='', logger=None):
    assert len(prob_grid.shape) == 2 and prob_grid.shape == class_grid.shape \
        and prob_grid.shape[0] > 2 and prob_grid.shape[1] > 2

    pdb.set_trace()
    h, w = prob_grid.shape
    voting_grid = -1*np.ones([h-2, w-2])
    for i in range(h-2):
        for j in range(w-2):
            tprob = prob_grid[i:i+3, j:j+3]
            tcls = class_grid[i:i+3, j:j+3]
            idx = np.argmin(tprob)

            voting_grid[i, j] = tcls[idx//3, idx % 3]
            for k in range(9):
                if tcls[k//3, k % 3] != voting_grid[i, j]:
                    voting_grid[i, j] = -1
                    break
                else:
                    continue

    # pdb.set_trace()
    freq = dict()
    for i in range(h-2):
        for j in range(w-2):
            val = int(voting_grid[i, j])
            if val == -1:
                continue
            else:
                if val in freq:
                    freq[val] += 1
                else:
                    freq[val] = 1
    if logger:
        logger.info('freq: {}'.format(
            sorted(freq.items(), key=lambda kv: kv[1], reverse=True)))
    else:
        print('freq: ', sorted(freq.items(),
                               key=lambda kv: kv[1], reverse=True))
    if len(freq.keys()) == 1:
        return list(freq.keys())[0], voting_grid
    elif len(freq.keys()) == 2:
        keys = list(freq.keys())
        if freq[keys[0]] > freq[keys[1]]:
            return keys[1], voting_grid
        else:
            return keys[0], voting_grid
    return -1, voting_grid


def detect_patch(vote_grid):
    has_patch = "False"
    original_cls = -1
    # pdb.set_trace()
    vote_list = list(Counter(vote_grid.reshape(-1)))
    if -1 in vote_list:
        vote_list.remove(-1)
    if len(vote_list) >= 2:
        has_patch = "True" 
        original_cls = vote_list[-1]
    else:
        has_patch = "False"
        original_cls = vote_list[0]
    return has_patch, original_cls


if __name__ == '__main__':
    calc_num = 0

    config = Configuration()

    # Set logger to record the process
    logger = get_mylogger('./log/detect_mrd/ResNet18_%d' % config.PATCH_SIZE)
    logger.info("Parameters: %s" % (config.__dict__))

    # Get images
    fimages = collect_images(config.FILEDIR, config.FILENUM)

    # Load model
    # .eval() is indispensable, otherwise the result goes wrong.
    model = models.__dict__[config.ARCH]().cuda().eval()
    model = torch.nn.DataParallel(model)

    model_file = './result/models/{}_{}/model_best.pth.tar'.format(
        config.ARCH, config.PATCH_SIZE)
    model.load_state_dict(torch.load(config.MODEL_NAME)['state_dict'])
    model = model.module

    # Detect
    keys = ['benign', 'patch']
    np.set_printoptions(threshold=np.inf)

    total_num = 0
    detect_fp, detect_fn, detect_tp, detect_tn = 0, 0, 0, 0
    restore_patch, restore_benign = 0, 0
    tmp_num = 0
    for i in trange(config.FILENUM):
        # get those classified correctly images
        fimg = os.path.join(config.FILEDIR, fimages['benign'][i])
        benign_img = torch.load(fimg[:-4]+'.pth').cuda()
        output = model(benign_img)
        prob, benign_cls = torch.max(output, 1)
        prob, benign_cls = prob.cpu().detach().numpy(), benign_cls.cpu().detach().numpy()

        # GoundTruth
        true_label = int(re.split('_|\.', fimages['benign'][i])[-2])
        if true_label != benign_cls:
            # continue
            pass

        # 删去label 5
        if true_label == 5:
            # continue
            pass
        # get images that can attack precisely
        fimg = os.path.join(config.FILEDIR, fimages['patch'][i])
        poisoned_img = torch.load(fimg[:-4]+'.pth').cuda()
        output = model(poisoned_img)
        prob, poisoned_cls = torch.max(output, 1)
        prob, poisoned_cls = prob.cpu().detach().numpy(), poisoned_cls.cpu().detach().numpy()
        if poisoned_cls != config.TARGET_LABLE:
            # continue
            pass
        total_num += 1

        for key in keys:
            fimg = os.path.join(config.FILEDIR, fimages[key][i])
            mdr_file = torch.load(fimg[:-4] + '_mrd.pth')

            # Predicted label
            if key == "benign":
                img = benign_img
                cls = benign_cls
            else:
                img = poisoned_img
                cls = poisoned_cls
            unoccluded_cls = cls
            cls_grid = mdr_file['cls']
            prob_grid = mdr_file['prob']
            vote_grid = mdr_file['vote']
            # _, vote_grid = mdr_vote(prob_grid, cls_grid)
            has_patch, original_cls = detect_patch(vote_grid)

            # pdb.set_trace()
            if key == "patch" and has_patch == "True":
                detect_tp += 1
            if key == "patch" and has_patch == "False":
                detect_fn += 1
            if key == "benign" and has_patch == "True":
                detect_fp += 1
            if key == "benign" and has_patch == "False":
                detect_tn += 1

            if key == "patch" and original_cls == true_label:
                restore_patch += 1
            if key == "benign" and original_cls == true_label:
                restore_benign += 1
                
    logger.info("Mdr method")
    logger.info("Used Image Quantity: %d" % total_num)
    logger.info("Detect tp:%.3f fn:%.3f fp:%.3f tn:%.3f" %
                (detect_tp/total_num, detect_fn/total_num, detect_fp/total_num, detect_tn/total_num))
    logger.info("Restore patch:%.3f benign:%.3f" %
                (restore_patch/total_num, restore_benign/total_num))
    # logger.info("Candidate num:(nms: %s):%d" % (config.NMS, calc_num))
    # logger.info("[patch,benign]Method1:%s,Method2:%s,Method3:%s" %
    #             (config.method1, config.method2, config.method3))
