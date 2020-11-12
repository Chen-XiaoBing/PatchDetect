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
        self.PATCH_SIZE = 10
        self.FILENUM = 1000
        self.FILEDIR = './result/attack/image_specific/ResNet18/ResNet18_{}/'.format(
            self.PATCH_SIZE)
        self.MODEL_NAME = './result/models/{}_{}/model_best.pth.tar'.format(
            self.ARCH, self.PATCH_SIZE)
        self.TOPK = 14
        self.TOPK_ALPHA1 = 0.95
        self.BOX_SIZE = self.PATCH_SIZE + 2


def euclidean_dist(x, y):
    return math.sqrt(pow(x[0]-y[0], 2), pow(x[1]-y[1], 2))


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


def detect_patch(topk_map, mdr_file, unoccluded_cls, config):
    
    cls_grid = mdr_file['cls']

    _, h, w = out.size()
    val, idx = out.view(-1).max(0)
    val = val.cpu().detach().numpy()
    idx = idx.cpu().detach().numpy()
    idx = [idx//w, idx % w]

    has_patch = "False"
    original_cls = unoccluded_cls
    if val >= config.TOPK_ALPHA1 * config.TOPK:
        # pdb.set_trace()
        occl_cls = cls_grid[idx[0]][idx[1]]
        if occl_cls == unoccluded_cls:
            has_patch = "False"
            original_cls = unoccluded_cls
        else:
            has_patch = "True"
            original_cls = occl_cls
    # else:
    #     # has_patch = 'else'
    #     _, idx = torch.topk(out.view(-1),5)
    #     # pdb.set_trace()
    #     idx = idx.cpu().detach().numpy()
    #     idx = [idx//w, idx % w]    
        
    #     occl_cls = []
    #     for i in range(len(idx[0])):
    #         # pdb.set_trace()
    #         occl_cls.append(cls_grid[idx[0][i]][idx[1][i]])
    #     collect_words = Counter(occl_cls)
        
    #     # all the occluded images predict the same label;
    #     # so the image is benign.
    #     if len(collect_words.keys()) == 1:
    #         has_patch = "False"
    #         original_cls = unoccluded_cls
    #     else:
    #         has_patch = "True"
    #         # 取出现次数最少的标签
    #         original_cls = list(collect_words.keys())[-1]
    return has_patch, original_cls


if __name__ == '__main__':

    config = Configuration()

    # Set logger to record the process
    logger = get_mylogger('./log/detect_topk/ResNet18_%d' % config.PATCH_SIZE)

    # Get images
    fimages = collect_images(config.FILEDIR, config.FILENUM)

    # Load model
    # TODO: .eval() is indispensable, otherwise the result goes wrong.
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
    detect_num = 0
    restore_num = 0
    tmp_num = 0
    for i in trange(config.FILENUM):
        # get those classified correctly images

        fimg = os.path.join(config.FILEDIR, fimages['benign'][i])
        img = torch.load(fimg[:-4]+'.pth').cuda()
        output = model(img)
        prob, cls = torch.max(output, 1)
        prob, cls = prob.cpu().detach().numpy(), cls.cpu().detach().numpy()
        true_label = int(re.split('_|\.', fimages['benign'][i])[-2])
        if true_label != cls:
            continue
        total_num += 1

        for key in keys:
            fimg = os.path.join(config.FILEDIR, fimages[key][i])
            img = torch.load(fimg[:-4]+'.pth').cuda()
            mdr_file = torch.load(fimg[:-4] + '_mdr.pth')

            # Predicted label
            output = model(img)
            _, cls = torch.max(output, 1)
            unoccluded_cls = cls.cpu().detach().numpy()

            # pdb.set_trace()
            # GoundTruth
            true_label = int(re.split('_|\.', fimg)[-2])

            conv1 = model.bn1(model.conv1(img))
            conv2d = torch.sum(conv1.squeeze(dim=0), dim=0)
            h, w = conv2d.size()
            _, idx = torch.topk(conv2d.view(-1), config.TOPK)
            conv2d = conv2d.view(-1)
            conv2d[:] = 0
            conv2d[idx] = 1
            conv2d = conv2d.view(1, h, w)
            out = torch.nn.AvgPool2d(
                (config.BOX_SIZE, config.BOX_SIZE), stride=(1, 1))(conv2d)
            # Recover from average value
            out = out * config.BOX_SIZE * config.BOX_SIZE
            has_patch, original_cls = detect_patch(
                out, mdr_file, unoccluded_cls, config)

            # pdb.set_trace()
            if key == "patch" and has_patch == "True":
                detect_num += 1
            if key == "patch" and original_cls == true_label:
                restore_num += 1
            if key == "patch" and has_patch == 'else':
                tmp_num += 1
    logger.info("Detect Rate: %.3f \n Restore Rate: %.3f" %
                (detect_num/total_num, restore_num/total_num))
    logger.info("%d %d" % (total_num,tmp_num))
