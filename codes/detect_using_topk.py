import logging
import os
import pdb
import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

import models
from mylogger import get_mylogger


class Configuration():
    def __init__(self):
        self.ARCH = "ResNet18"
        self.PATCH_SIZE = 7
        self.FILENUM = 10000
        self.FILEDIR = './result/attack/image_specific/ResNet18/ResNet18_{}/'.format(
            self.PATCH_SIZE)
        self.MODEL_NAME = './result/models/{}_{}/model_best.pth.tar'.format(
            self.ARCH, self.PATCH_SIZE)
        self.TOPK = 10
        self.TOPK_ALPHA1 = 0.8
        self.TOPK_ALPHA2 = 0.3
        self.BOX_SIZE = self.PATCH_SIZE
        self.TARGET_LABLE = 5
        self.NMS = True

        # 【patch,benign]
        self.method1 = [0, 0]
        self.method2 = [0, 0]
        self.method3 = [0, 0]


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


def nms(idx, config, threshold=0.4):
    patch_size = config.PATCH_SIZE
    if len(idx) == 1:
        return idx
    else:
        nms_list = []
        nms_list.append(idx[1])
        for i in range(1, len(idx)):
            should_be_stored = 0
            for j in range(len(nms_list)):
                delta_x = abs(idx[i][0] - nms_list[j][0])
                delta_y = abs(idx[i][1] - nms_list[j][1])
                if nms_list[j] != idx[i] and (delta_x >= patch_size or delta_y >= patch_size):
                    should_be_stored += 1
                    continue
                overlap_area = (patch_size - delta_x) * (patch_size - delta_y)
                overlap_ratio = overlap_area / (patch_size * patch_size)
                if nms_list[j] != idx[i] and overlap_ratio <= threshold:
                    should_be_stored += 1
            # pdb.set_trace()
            if should_be_stored == len(nms_list):
                nms_list.append(idx[i])
    return nms_list[:]


def detect_patch(topk_map, mdr_file, unoccluded_cls, config, key, fimg):

    cls_grid = mdr_file['cls']

    _, _, w = out.size()
    val, idx = out.view(-1).max(0)
    val = val.cpu().detach().numpy()
    idx = idx.cpu().detach().numpy()
    idx = [idx//w, idx % w]

    has_patch = "False"
    original_cls = unoccluded_cls
    if val >= config.TOPK_ALPHA1 * config.TOPK:
        # if False:
        # pdb.set_trace()
        occl_cls = cls_grid[idx[0]][idx[1]]
        if occl_cls == unoccluded_cls:
            has_patch = "False"
            original_cls = unoccluded_cls
            config.method1[1] += 1
        else:
            has_patch = "True"
            original_cls = occl_cls
            config.method1[0] += 1

    elif val <= config.TOPK * config.TOPK_ALPHA2:
        has_patch = "False"
        original_cls = unoccluded_cls
        config.method2[1] += 1
    else:
        _, idx = torch.topk(out.view(-1), 14)
        idx = idx.cpu().detach().numpy()
        idx_list = []
        for i in range(len(idx)):
            idx_list.append((idx[i]//w, idx[i] % w))
        # deepcopy, if wants shallowcopy, use `idx=idx_list`
        idx = idx_list[:]

        # pdb.set_trace()
        if config.NMS == True:
            idx = nms(idx, config)

        global calc_num
        calc_num += len(idx)

        occl_cls = []
        for i in range(len(idx)):
            # pdb.set_trace()
            occl_cls.append(cls_grid[idx[i][0]][idx[i][1]])
        collect_words = Counter(occl_cls)

        if key == "patch":
            # pdb.set_trace()
            pass

        # all the occluded images predict the same label;
        # so the image is benign.
        if len(collect_words.keys()) == 1 and list(collect_words)[0] == unoccluded_cls:
            if key == "patch":
                # pdb.set_trace()
                pass
            has_patch = "False"
            original_cls = unoccluded_cls
            config.method3[1] += 1
        else:
            has_patch = "True"
            # 取出现次数最少的标签
            original_cls = list(collect_words.keys())[-1]
            config.method3[0] += 1

    return has_patch, original_cls


if __name__ == '__main__':

    calc_num = 0

    config = Configuration()

    # Set logger to record the process
    logger = get_mylogger('./log/detect_topk/ResNet18_%d' % config.PATCH_SIZE)
    logger.info("Parameters: %s" % (config.__dict__))

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
            continue

        # 删去label 5
        if true_label == 5:
            continue
        # get images that can attack precisely
        fimg = os.path.join(config.FILEDIR, fimages['patch'][i])
        poisoned_img = torch.load(fimg[:-4]+'.pth').cuda()
        output = model(poisoned_img)
        prob, poisoned_cls = torch.max(output, 1)
        prob, poisoned_cls = prob.cpu().detach().numpy(), poisoned_cls.cpu().detach().numpy()
        if poisoned_cls != config.TARGET_LABLE:
            continue
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

            # pdb.set_trace()
            # # GoundTruth
            # true_label = int(re.split('_|\.', fimg)[-2])

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
                out, mdr_file, unoccluded_cls, config, key, fimg)

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

    logger.info("Used Image Quantity: %d" % total_num)
    logger.info("Detect tp:%.3f fn:%.3f fp:%.3f tn:%.3f" %
                (detect_tp/total_num, detect_fn/total_num, detect_fp/total_num, detect_tn/total_num))
    logger.info("Restore patch:%.3f benign:%.3f" %
                (restore_patch/total_num, restore_benign/total_num))
    logger.info("Candidate num:(nms: %s):%d" % (config.NMS, calc_num))
    logger.info("[patch,benign]Method1:%s,Method2:%s,Method3:%s" %
                (config.method1, config.method2, config.method3))
