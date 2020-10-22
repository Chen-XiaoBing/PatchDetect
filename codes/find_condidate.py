import os
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import guided_propagate


def get_locate_model(arch, ckpt_path):
    if arch == 'resnet50':
        locate_model = guided_propagate.resnet50().cuda()
    elif arch == 'resnet152':
        locate_model = guided_propagate.resnet152().cuda()
    elif arch == 'mobilenet_v2':
        locate_model = guided_propagate.mobilenet_v2().cuda()
    else:
        raise Exception('Unsupport arch: ', arch)

    ckpt = torch.load(ckpt_path)
    if torch.cuda.device_count() > 1:
        locate_model= torch.nn.DataParallel(locate_model)
    locate_model.load_state_dict(ckpt['state_dict'])
    return locate_model

def get_feature_importance_map(model, image, label):
    assert image.size()[0] == 1
    image.requires_grad = True

    output = model(image)
    mask = torch.full(output.size(), 0).cuda()
    mask[0, label] = 1
    output = output * mask
    criterion = nn.CrossEntropyLoss().cuda()
    label = torch.LongTensor(np.array([label])).cuda()
    loss = criterion(output, label)
    loss.backward()
    temperature = torch.sum(image.grad, 1)

    fm_size = image_size-patch_size+1
    fm = np.zeros((fm_size, fm_size))
    for h in range(fm_size):
        for w in range(fm_size):
            ts = torch.sum(temperature[0, h:h+patch_size, w:w+patch_size])
            fm[h,w] = ts

    return fm


image_size = 224
patch_rate = 0.1
arch = 'resnet50'
ckpt = './data/models/resnet50_0.1/model_best.pth.tar'
input_dir = 'data/attack_pic_2/'
cudnn.benchmark = True
model = get_locate_model(arch, ckpt).eval()
patch_size = int(math.sqrt(image_size*image_size*patch_rate))
files = os.listdir(input_dir)
files = list(filter(lambda x : x.endswith('.pth') \
        and not x.endswith('mdr.pth') \
        and not x.endswith('fm.pth'), files))
for i in range(len(files)):
    img_file = files[i]
    print('{}/{} {}'.format(i, len(files), img_file))
    if img_file.startswith('ori'): label = 859
    else:                          label = int(img_file[:-4].split('_')[-1])
    img_file = os.path.join(input_dir, img_file)
    img = torch.load(img_file)
    outfile = img_file[:-4]+'_fm.pth'
    fm = get_feature_importance_map(model, img, label)
    torch.save(fm, outfile)
