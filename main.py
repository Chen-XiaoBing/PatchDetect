from utils import Graph, calc_total_flops, calc_overlap_ratio, retrieve_net_info
from detect import mdr_test
import re
import argparse
import numpy as np
import csv

import torch
from torchvision import models

mdr_test()
exit(0)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--patch_ratio', type=float, default='0.05')
args = parser.parse_args()


# model_names = sorted(name for name in models.__dict__ if
#                      name.islower() and not name.startswith("__") # and "inception" in name
#                      and callable(models.__dict__[name]))

names = [
'alexnet',
'densenet121',
'densenet161',
'densenet169',
'densenet201',
'resnet101',
'resnet152',
'resnet18',
'resnet34',
'resnet50',
'vgg11',
'vgg11_bn',
'vgg13',
'vgg13_bn',
'vgg16',
'vgg16_bn',
'vgg19',
'vgg19_bn']

items = []
for name in names:
    for patch_ratio in range(3, 11):
        name = 'vgg16'
        print('processing model {}, ratio {}'.format(name, patch_ratio/100.))
        # model = getattr(models, args.model)()
        model = getattr(models, name)()
        if 'inception' in name:
            input = torch.zeros([1, 3, 299, 299])
        else:
            input = torch.zeros([1, 3, 224, 224])

        g = retrieve_net_info(model, input, verbose=True)
        # flops = calc_total_flops(g)
        # print('model {}, total flops: {} Gflops'.format(name, flops / 1e9))
        # reuse_ratio = calc_overlap_ratio(g, patch_ratio/100.)
        # print("{}, {}, {}".format(name, patch_ratio/100., reuse_ratio))
        # items.append([name, patch_ratio/100., reuse_ratio])
        exit(0)

# with open('report.csv', 'w') as f:
#     f_csv = csv.writer(f)
#     f_csv.writerow(['model_name', 'patch_ratio', 'reuse_ratio'])
#     f_csv.writerows(items)
