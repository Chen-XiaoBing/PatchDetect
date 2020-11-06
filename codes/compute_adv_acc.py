import re
import os
import pdb
import torch
import models
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
from glob import glob
from tqdm import tqdm, trange
from image_specific_attack import create_model



def load_image(image_path: str):
    filename = glob(image_path+"*.pth")
    adv_examples = []
    labels = []
    for file in tqdm(filename[0:10000]):
        adv_example = torch.load(file)
        adv_examples.append(adv_example)

        label = int(re.split('_|\.', file)[-2])
        labels.append(label)

    return adv_examples, labels



"""
patch-size=7 : attack_acc=0.541, normal_acc=0.9301
patch-size=10: attack_acc=0.827, normal_acc=0.9241
"""
if __name__ == '__main__':
    patch_size = 10
    print("patch_size:", patch_size)

    adv_example_path = 'result/attack/image_specific/ResNet18/ResNet18_{}/ori/'.format(
        patch_size)
    adv_examples, labels = load_image(adv_example_path)

    model = create_model(
        'ResNet18', './result/models/ResNet18_{}/model_best.pth.tar'.format(patch_size)).eval()

    attack_acc = 0
    for i in trange(len(adv_examples)):
        adv_example = adv_examples[i]
        label = labels[i]

        adv_out = F.log_softmax(model(adv_example), dim=1)
        adv_out_probs, adv_out_labels = adv_out.max(1)
        # have already set the target label to 5.
        if adv_out_labels == label:
            attack_acc += 1
    print("attack acc: ", attack_acc/len(adv_examples))
