import os
import random
import logging
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

class AttackConfig(object):
    def __init__(self, image_size, conf, logger, target, \
            max_count, test_size, train_size, batch_size):
        self.image_size = image_size
        self.conf = conf
        self.logger = logger
        self.target = target
        self.max_count = max_count
        self.test_size = test_size
        self.train_size = train_size
        self.batch_size = batch_size

def get_logger(filename):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(filename, mode='w')
    fh.setLevel(logging.INFO)

    ch=logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # formatter = logging.Formatter(
    #         "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    formatter = logging.Formatter(
            "%(asctime)s : %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)

def create_model(arch, ckpt):
    print("=> creating model ")
    model = models.__dict__[arch]()
    model.cuda()
    ckpt = torch.load(ckpt)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['state_dict'].items()})
    model.load_state_dict(ckpt['state_dict'])
    return model

def init_env():
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names,
                help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
    parser.add_argument('--seed', type=int, default=1338)
    parser.add_argument('--data', metavar='DIR', default='./data/train')
    parser.add_argument('--ckpt', type=str, default='./data/models/resnet50_0.05/model_best.pth.tar')
    parser.add_argument('--train_size', type=int, default=10000, help='Number of training images')
    parser.add_argument('--test_size', type=int, default=10000, help='Number of test images')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--patch-rate', default=0.05, type=float)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--target', type=int, default=859, help='The target class: 859 == toaster')
    parser.add_argument('--conf', type=float, default=0.9,
                        help='Stop attack on image when target classifier reaches this value for target class')
    # parser.add_argument('--max-count', type=int, default=30*256)
    parser.add_argument('--max-count', type=int, default=100*256)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    init_seed(args.seed)
    cudnn.benchmark = True
    return args

def get_dataloader(image_size, data_path, train_size, test_size):
    idx = np.arange(50000)
    np.random.shuffle(idx)
    train_idx = idx[:train_size]
    test_idx = idx[train_size:train_size + test_size]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder(
        data_path,
        transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, sampler=SubsetRandomSampler(train_idx),
        num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, sampler=SubsetRandomSampler(test_idx),
        num_workers=1, pin_memory=True)

    return train_loader, test_loader

def init_patch(image_size, patch_rate):
    image_size = image_size**2
    noise_size = image_size*patch_rate
    noise_dim = int(noise_size**(0.5))
    # patch = np.random.rand(1, 3, noise_dim, noise_dim)
    patch = np.zeros((1, 3, noise_dim, noise_dim)) + 0.5
    return patch

def square_transform(patch, data_shape, patch_size, image_size):
    padded_patch = np.zeros(data_shape)

    assert data_shape[0] == 1
    # random location
    rand_h = np.random.choice(image_size-patch_size+1)
    rand_w = np.random.choice(image_size-patch_size+1)
    # apply patch to dummy image
    padded_patch[0, :, rand_h:rand_h+patch_size, rand_w:rand_w+patch_size] = patch

    mask = np.copy(padded_patch)
    mask[mask != 0] = 1.0

    return padded_patch, mask, rand_h, rand_w

def submatrix(arr):
    x, y = np.nonzero(arr)
    # Using the smallest and largest x and y indices of nonzero elements,
    # we can find the desired rectangular bounds.
    # And don't forget to add 1 to the top bound to avoid the fencepost problem.
    return arr[x.min():x.max()+1, y.min():y.max()+1]

def add_zeromask(images, patch_size):
    _, _, height, width = images.size()
    masked_images = images.clone()
    hp = random.randint(0, height-patch_size+1)
    wp = random.randint(0, width-patch_size+1)
    # occluded range
    hstart = max(0, hp-1)
    hend   = min(height-1, hp+patch_size)
    wstart = max(0, wp-1)
    wend   = min(width-1, wp+patch_size)
    masked_images[:, :, hstart:hend+1, wstart:wend+1] = 0
    return masked_images

def main():
    if False:
        min_val, max_val = 10., 0.
        min_val, max_val = get_input_range('data/train/', min_val, max_val)
        min_val, max_val = get_input_range('data/val/', min_val, max_val)
        print(min_val, max_val)

    args = init_env()
    logger = get_logger('log/attack/image_specific_{}_{}'.format(args.arch, args.patch_rate))
    model = create_model(args.arch, args.ckpt).cuda().eval()
    _, data_loader = get_dataloader(
            args.image_size, args.data, args.train_size, args.test_size)

    config = AttackConfig(args.image_size, args.conf, logger,
            args.target, args.max_count, args.test_size, args.train_size, args.batch_size)

    for i, (image, label) in enumerate(data_loader):
        patch = init_patch(args.image_size, args.patch_rate)
        data_shape = image.size()
        patch_size = patch.shape[-1]
        image_size = args.image_size

        padded_patch, patch_mask, rand_h, rand_w = \
                square_transform(patch, data_shape, patch_size, image_size)
        padded_patch, patch_mask = \
                torch.FloatTensor(padded_patch).cuda(), torch.FloatTensor(patch_mask).cuda()
        image = image.cuda()

        iter_num=0
        batch_size = config.batch_size
        while iter_num < config.max_count/batch_size:
            adv_image = torch.mul(1-patch_mask, image) + torch.mul(patch_mask, padded_patch)
            adv_image = torch.clamp(adv_image, -2.117904, 2.64)
            adv_image = Variable(adv_image.data, requires_grad=True)

            expand_adv_image = adv_image.expand([batch_size, 3, 224, 224])

            occlude_mask = np.ones([batch_size, 3, 224, 224])
            for bidx in range(batch_size):
                hp = random.randint(0, image_size-2*patch_size+1)
                if hp > rand_h-patch_size/2: hp += patch_size
                wp = random.randint(0, image_size-2*patch_size+1)
                if wp > rand_w-patch_size/2: wp += patch_size
                # hp = random.randint(0, image_size-patch_size+1)
                # wp = random.randint(0, image_size-patch_size+1)

                # occluded range
                hstart, hend = max(0, hp-1), min(image_size-1, hp+patch_size)
                wstart, wend = max(0, wp-1), min(image_size-1, wp+patch_size)

                occlude_mask[bidx, :, hstart:hend+1, wstart:wend+1] = 0
            occlude_mask = torch.FloatTensor(occlude_mask).cuda()
            occlude_image = torch.mul(occlude_mask, expand_adv_image)

            adv_out = F.log_softmax(model(occlude_image))
            adv_out_probs, adv_out_labels = adv_out.max(1)
            loss = -torch.sum(adv_out[:, 859])
            loss.backward()
            adv_grad = adv_image.grad.clone()
            adv_image.grad.data.zero_()
            padded_patch -= adv_grad
            iter_num+=1
            if iter_num % 10 == 0:
                logger.info('image idx: {}, iter: {}, loss: {}'.format(i, iter_num, loss))

        saved_path = './data/attack/image_specific/ori_{}_{}_{}_{}.pth'.\
                format(args.arch, args.patch_rate, i, label.numpy()[0])
        torch.save(image.data, saved_path)
        adv_image = torch.mul(1-patch_mask, image) + torch.mul(patch_mask, padded_patch)
        adv_image = torch.clamp(adv_image, -2.117904, 2.64)
        saved_path = './data/attack/image_specific/patched_{}_{}_{}_{}.pth'.\
                format(args.arch, args.patch_rate, i, label.numpy()[0])
        torch.save(adv_image.data, saved_path)

        adv_image = torch.mul(1-patch_mask, image) + torch.mul(patch_mask, padded_patch)
        adv_image = torch.clamp(adv_image, -2.117904, 2.64)
        saved_path = './data/attack/image_specific/patched_{}_{}_{}_{}.jpg'.\
                format(args.arch, args.patch_rate, i, label.numpy()[0])
        torchvision.utils.save_image(adv_image.data, saved_path)

        saved_path = './data/attack/image_specific/ori_{}_{}_{}_{}.jpg'.\
                format(args.arch, args.patch_rate, i, label.numpy()[0])
        torchvision.utils.save_image(image.data, saved_path)
        adv_image = torch.mul(1-patch_mask, image) + torch.mul(patch_mask, padded_patch)
        adv_image = torch.clamp(adv_image, -2.117904, 2.64)
        saved_path = './data/attack/image_specific/patched_{}_{}_{}_{}.jpg'.\
                format(args.arch, args.patch_rate, i, label.numpy()[0])
        torchvision.utils.save_image(adv_image.data, saved_path)


if __name__ == '__main__':
    main()

'''
python utils/image_specific_attck.py --arch resnet50 --ckpt data/models/resnet50_0.1/model_best.pth.tar --patch-rate 0.1 --batch-size 128
'''
