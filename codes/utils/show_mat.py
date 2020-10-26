import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

def show_and_save(image=None, infile=None, outfile=None, show=True, save=True, add_thre=False):
    if infile != None:
        content = list()
        filename = infile
        f = open(filename, 'r')
        reader = csv.reader(f)
        for row in reader:
            content.append(row)

        f.close()
        content = np.array(content, dtype=np.float)

    if image.all() != None:
        content = image

    if add_thre:
        # thea = content.sum()/content.size
        # content = (content > thea).astype(int)
        content = 100 * content
    plt.matshow(content)
    if show:
        plt.show()
    if save and outfile != None:
	    plt.savefig(outfile)


def save_image():
    filedir = 'data/attack_pic'
    files = os.listdir(filedir)
    files = filter(lambda x : x.endswith('.pth')
                              and not x.endswith('mdr.pth')
                              and not x.endswith('fm.pth'),
                   files)
    for i in files:
        print(i)
        img = torch.load(os.path.join(filedir, i), map_location=torch.device('cpu')).numpy()
        torchvision.utils.save_image(torch.FloatTensor(img), os.path.join(filedir, i)[:-4]+'.jpg', normalize=False)
