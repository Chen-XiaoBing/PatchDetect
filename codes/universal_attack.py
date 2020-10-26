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
    def __init__(self, image_size, conf, logger, target, max_count, test_size, train_size):
        self.image_size = image_size
        self.conf = conf
        self.logger = logger
        self.target = target
        self.max_count = max_count
        self.test_size = test_size
        self.train_size = train_size

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
    parser.add_argument('--train_size', type=int, default=2000, help='Number of training images')
    parser.add_argument('--test_size', type=int, default=2000, help='Number of test images')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--patch-rate', default=0.05, type=float)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--target', type=int, default=859, help='The target class: 859 == toaster')
    parser.add_argument('--conf', type=float, default=0.9,
                        help='Stop attack on image when target classifier reaches this value for target class')
    parser.add_argument('--max_count', type=int, default=1000,
                        help='max number of iterations to find adversarial example')
    args = parser.parse_args()

    init_seed(args.seed)
    cudnn.benchmark = True
    return args

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
    patch = np.random.rand(1, 3, noise_dim, noise_dim)
    return patch

def square_transform(patch, data_shape, patch_size, image_size):
    padded_patch = np.zeros(data_shape)

    for i in range(padded_patch.shape[0]):
        # random location
        rand_h = np.random.choice(image_size-patch_size+1)
        rand_w = np.random.choice(image_size-patch_size+1)
        # apply patch to dummy image
        padded_patch[i, :, rand_h:rand_h+patch_size, rand_w:rand_w+patch_size] = patch

    mask = np.copy(padded_patch)
    mask[mask != 0] = 1.0

    return padded_patch, mask

def attack(model, images, padded_patch, mask, config):
    model.eval()
    x_out = F.softmax(model(images))
    target_prob = x_out.data[0][config.target]
    adv_images = torch.mul((1-mask), images) + torch.mul(mask, padded_patch)

    count = 0
    while config.conf > target_prob and count < config.max_count:
        adv_images = Variable(adv_images.data, requires_grad=True)
        adv_out = F.log_softmax(model(adv_images))
        adv_out_probs, adv_out_labels = adv_out.max(1)
        loss = -adv_out[0][config.target]
        loss.backward()

        adv_grad = adv_images.grad.clone()
        adv_images.grad.data.zero_()
        padded_patch -= adv_grad

        adv_images = torch.mul((1-mask), images) + torch.mul(mask, padded_patch)
        adv_images = torch.clamp(adv_images, -2.117904, 2.64)

        out = F.softmax(model(adv_images))
        target_prob = out.data[0][config.target]
        count += 1

    return adv_images, mask, padded_patch

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

def train(model, patch, data_loader, config):
    patch_shape = patch.shape
    total, norm_total, success = 0, 0, 0
    for batch_idx, (images, labels) in enumerate(data_loader):
        if torch.cuda.device_count() > 1:
            images, labels = images.cuda(), labels.cuda()
        images, labels = Variable(images), Variable(labels)

        masked_images = add_zeromask(images, patch.shape[-1])
        pred = model(masked_images)
        total += 1
        if pred.data.max(1)[1][0] == labels.data[0]: norm_total += 1

        data_shape = images.data.cpu().numpy().shape
        padded_patch, mask = square_transform(
                patch, data_shape, patch.shape[-1], config.image_size)
        padded_patch, mask = torch.FloatTensor(padded_patch), torch.FloatTensor(mask)
        if torch.cuda.device_count() > 1:
            padded_patch, mask = padded_patch.cuda(), mask.cuda()
        padded_patch, mask = Variable(padded_patch), Variable(mask)
        adv_images, mask, padded_patch = attack(
                model, images, padded_patch, mask, config)

        adv_label = model(adv_images).data.max(1)[1][0]
        ori_label = labels.data[0]
        if adv_label == config.target:
            success += 1
        masked_patch = torch.mul(mask, padded_patch)
        patch = masked_patch.data.cpu().numpy()

        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]):
            for j in range(new_patch.shape[1]):
                new_patch[i][j] = submatrix(patch[i][j])
        patch = new_patch
        if batch_idx % 100 == 99:
            config.logger.info('Train: success: {}, normal: {}, total: {}'.format(success, norm_total, total))
        # torchvision.utils.save_image(images.data, 'data/attack/{}_ori.png'.format(batch_idx), normalize=False)
        # torchvision.utils.save_image(adv_images.data, 'data/attack/{}_adv.png'.format(batch_idx), normalize=False)

    config.logger.info("Train: Normal examples acc: %.3f" % (norm_total/total))
    config.logger.info("Train: Attack success rate: %.3f" % (success/total))
    return patch

def test(model, patch, test_loader, config):
    model.eval()
    patch_shape = patch.shape
    total, norm_total, success = 0, 0, 0
    for batch_idx, (images, labels) in enumerate(test_loader):
        if torch.cuda.device_count() > 1:
            images, labels = images.cuda(), labels.cuda()


        masked_images = add_zeromask(images, patch.shape[-1])
        pred = model(masked_images)
        total += 1
        if pred.data.max(1)[1][0] == labels.data[0]: norm_total += 1

        data_shape = images.data.cpu().numpy().shape
        padded_patch, mask = square_transform(patch, data_shape, patch.shape[-1], config.image_size)
        padded_patch, mask = torch.FloatTensor(padded_patch), torch.FloatTensor(mask)
        if torch.cuda.device_count() > 1:
            padded_patch, mask = padded_patch.cuda(), mask.cuda()
        adv_images = torch.mul((1-mask), images) + torch.mul(mask, padded_patch)
        adv_images = torch.clamp(adv_images, -2.117904, 2.64)
        adv_label = model(adv_images).data.max(1)[1][0]
        if adv_label == config.target: success += 1
        if batch_idx % 100 == 99:
            config.logger.info('Test: success: {}, normal: {}, total: {}'.format(success, norm_total, total))
    config.logger.info("Test: Normal examples acc: %.3f" % (norm_total/total))
    config.logger.info("Test: Attack success rate: %.3f" % (success/total))

def get_input_range(path, min_val, max_val):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(path, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=512, shuffle=False,
        num_workers=4, pin_memory=True)
    min_val, max_val = 10., -5.
    for i, (image, _) in enumerate(test_loader):
        image = image.cuda()
        max_val = max(torch.max(image).cpu().numpy(), max_val)
        min_val = min(torch.min(image).cpu().numpy(), min_val)
        if i % 1000 == 0:
            print(min_val, max_val)

    return min_val, max_val

def main():
    if False:
        min_val, max_val = 10., 0.
        min_val, max_val = get_input_range('data/train/', min_val, max_val)
        min_val, max_val = get_input_range('data/val/', min_val, max_val)
        print(min_val, max_val)

    args = init_env()
    logger = get_logger('log/attack/universal_{}_{}'.format(args.arch, args.patch_rate))
    model = create_model(args.arch, args.ckpt)
    train_loader, test_loader = get_dataloader(
            args.image_size, args.data, args.train_size, args.test_size)
    patch = init_patch(args.image_size, args.patch_rate)

    config = AttackConfig(args.image_size, args.conf, logger,
            args.target, args.max_count, args.test_size, args.train_size)

    for epoch in range(1, args.epochs + 1):
        logger.info('epoch: {}'.format(epoch))
        patch = train(model, patch, train_loader, config)
        os.makedirs("data/patch/{}_{}_patch".format(args.arch, args.patch_rate), exist_ok=True)
        torch.save({'patch': patch, 'epoch': epoch},
                   "data/patch/{}_{}_patch/{}_patch.pth".format(args.arch, args.patch_rate, epoch))
        test(model, patch, test_loader, config)


def main():


if __name__ == '__main__':
    main()

'''
python utils/universal_attack.py --arch resnet50 --data ./data/train --ckpt ./data/models/resnet50_0.05/model_best.pth.tar
'''
