import os
import math
import logging
import numpy as np

import torch
import torchvision
import torch.nn.functional as F
import models
from mylogger import get_mylogger


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

if __name__ == '__main__':
    logger = get_mylogger("./log/detect_topk")

    arch = 'ResNet18'
    patch_size = 10
    filenum = 10000
    filedir = './result/attack/image_specific/ResNet18/ResNet18_10/'

    fimages = collect_images(filedir, filenum)

    assert arch == 'ResNet18'
    model = models.__dict__[arch]().cuda().eval()

    model = torch.nn.DataParallel(model)
    model_file = './result/models/{}_{}/model_best.pth.tar'.format(
        arch, patch_size)
    model.load_state_dict(torch.load(model_file)['state_dict'])
    model = model.module

    keys = ['benign', 'patch']
    k_list = [(i+5) for i in range(50)] 
    np.set_printoptions(threshold=np.inf)
    box_size = 12
    for k in k_list:
        count = {'benign': np.zeros(filenum), 'patch': np.zeros(filenum)}
        for i in range(filenum):
            for key in keys:
                fimg = os.path.join(filedir, fimages[key][i])
                img = torch.load(fimg[:-4]+'.pth')
                conv1 = model.bn1(model.conv1(img))
                conv2d = torch.sum(conv1.squeeze(dim=0), dim=0)
                h, w = conv2d.size()

                _, idx = torch.topk(conv2d.view(-1), k)
                conv2d = conv2d.view(-1)
                conv2d[:] = 0
                conv2d[idx] = 1
                conv2d = conv2d.view(1, h, w)
                out = torch.nn.AvgPool2d(
                    (box_size, box_size), stride=(1, 1))(conv2d)
                _, h, w = out.size()
                val, idx = out.view(-1).max(0)
                val = val.cpu().detach().numpy()*box_size*box_size
                idx = idx.cpu().detach().numpy()
                idx = [idx//w, idx % w]
                # logger.info(key, filenum, idx, val)
                count[key][i] = round(val)
        bins = [i*k/30 for i in range(31)]
        benign_hist, _ = np.histogram(count['benign'], bins)
        patch_hist,  _ = np.histogram(count['patch'], bins)
        logger.info(k)
        logger.info('bins:   '+' '.join('{:4d}'.format(int(i)) for i in bins[1:]))
        logger.info('benign: '+' '.join('{:4d}'.format(i) for i in benign_hist))
        logger.info('patch:  '+' '.join('{:4d}'.format(i) for i in patch_hist))
