import os
import math
import argparse
import numpy as np
import logging

import pdb
import csv
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
# import torchvision.models as models
import models
from torch.utils.data.sampler import SubsetRandomSampler
from detect.occluded_vgg import mdr_vote
from mylogger import get_mylogger
from tqdm import tqdm


def init_args():
    model_names = ['modebilenet_v2', 'resnet50', 'resnet152', 'ResNet18']
    parser = argparse.ArgumentParser(description='MDR testing')
    parser.add_argument('--patch-size', default=7, type=int,
                        help='patch to dataset')
    parser.add_argument('--arch', metavar='ARCH', default='ResNet18',
                        choices=model_names, help='model architecture: ' +
                        ' | '.join(model_names) + ' (default: ResNet18)')
    parser.add_argument('--ckpt', type=str,
                        default='./result/models/ResNet18_7/model_best.pth.tar')
    args = parser.parse_args()
    return args


def gen_model(arch, ckpt):
    model = models.__dict__[arch]().cuda()
    ckpt = torch.load(ckpt)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(ckpt['state_dict'])
    cudnn.benchmark = True
    return model


def run_mrd(model, image, patch_size, logger):
    image = image.cuda()

    image_size = image.size(3)
    h_range = w_range = image_size - patch_size + 1

    prob_grid = np.zeros([h_range, w_range])
    class_grid = np.zeros([h_range, w_range])

    occl_mask_shape = list(image.size())
    occl_mask_shape[0] = w_range
    images = image.expand(occl_mask_shape)
    
    for h in range(h_range):
        occl_mask = np.ones(occl_mask_shape)
        for w in range(w_range):
            hstart = max(0, h-1)
            hend = min(image_size-1, h+patch_size)
            wstart = max(0, w-1)
            wend = min(image_size-1, w+patch_size)
            occl_mask[w, :, hstart:hend+1, wstart:wend+1] = 0
        occl_mask = torch.FloatTensor(occl_mask).cuda()
        masked_image = torch.mul(occl_mask, images)
        output = model(masked_image)
        prob, cls = torch.max(output, 1)
        prob, cls = prob.cpu().detach().numpy(), cls.cpu().detach().numpy()
        
        # 一次处理一列数据
        prob_grid[h, :], class_grid[h, :] = prob, cls
    cls, voting_grid = mdr_vote(prob_grid, class_grid, 0, '', logger)
    return prob_grid, class_grid, voting_grid, cls


def main():
    args = init_args()
    logger = get_mylogger(
        'log/image_detect_{}_{}'.format(args.arch, args.patch_size))
    model = gen_model(args.arch, args.ckpt).eval()

    total_nr, succ_nr = 0, 0

    file_dir = 'result/attack/image_specific/ResNet18/ResNet18_7/'
    os.makedirs(file_dir, exist_ok=True)
    # pdb.set_trace()
    files = [f for f in os.listdir(file_dir)]
    for f in files:
        # ori_resnet50_0.1_65_37.pth
        if '_mrd' in f:
            continue

        try:
            has_patch, arch, patch_size, idx, label = (f[:-4]).split('_')
        except:
            pdb.set_trace()

        has_patch = (has_patch == 'patched')
        patch_size, idx, label = float(patch_size), int(idx), int(label)
        if not f.endswith('pth') or arch != args.arch or patch_size != args.patch_size:
            continue
        if f[:-4]+'_mrd.pth' in files:
            continue

        image = torch.load(os.path.join(file_dir, f))

        image_size = image.size()[-1]
        patch_size = args.patch_size

        prob_grid, class_grid, voting_grid, pred = run_mrd(
            model, image, patch_size, logger)
        torch.save({'prob': prob_grid, 'cls': class_grid, 'vote': voting_grid},
                   os.path.join(file_dir, f[:-4]+'_mrd.pth'))

        if pred == label:
            succ_nr += 1
        total_nr += 1

        logger.info("file: {}, label: {}, pred: {}, ({}/{})".format(f,
                                                                    label, pred, succ_nr, total_nr))


if __name__ == '__main__':
    main()

'''
python detect/image_specific_mrd.py --arch resnet50 --patch-size 0.1 --ckpt data/models/resnet50_0.1/model_best.pth.tar
'''
