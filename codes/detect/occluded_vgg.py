import time
import math
import torch
import numpy as np
import torch.nn as nn
from torch import  optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

class VGG16(nn.Module):
    def __init__(self, patch_size, fc_in=512, num_classes=10):
        super(VGG16, self).__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        def conv_bn_relu(f_in, f_out, kernel_size, padding):
            return nn.Sequential(
                nn.Conv2d(f_in, f_out, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(f_out),
                nn.ReLU(True),
            )

        self.conv1 = conv_bn_relu(3, 64, 3, 1)
        self.conv2 = conv_bn_relu(64, 64, 3, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3 = conv_bn_relu(64, 128, 3, 1)
        self.conv4 = conv_bn_relu(128, 128, 3, 1)
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv5 = conv_bn_relu(128, 256, 3, 1)
        self.conv6 = conv_bn_relu(256, 256, 3, 1)
        self.conv7 = conv_bn_relu(256, 256, 3, 1)
        self.pool7 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv8 = conv_bn_relu(256, 512, 3, 1)
        self.conv9 = conv_bn_relu(512, 512, 3, 1)
        self.conv10 = conv_bn_relu(512, 512, 3, 1)
        self.pool10 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv11 = conv_bn_relu(512, 512, 3, 1)
        self.conv12 = conv_bn_relu(512, 512, 3, 1)
        self.conv13 = conv_bn_relu(512, 512, 3, 1)
        self.pool13 = nn.MaxPool2d(kernel_size=2,stride=2)
        # print('model fc in: ', fc_in)
        self.classifier = nn.Sequential(
            #14
            nn.Linear(fc_in,4096),
            nn.ReLU(True),
            nn.Dropout(),
            #15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #16
            nn.Linear(4096,num_classes),

        )
        self.features = [
            self.conv1,
            self.conv2,
            self.pool2,
            self.conv3,
            self.conv4,
            self.pool4,
            self.conv5,
            self.conv6,
            self.conv7,
            self.pool7,
            self.conv8,
            self.conv9,
            self.conv10,
            self.pool10,
            self.conv11,
            self.conv12,
            self.conv13,
            self.pool13,
        ]

    def forward(self, x):
        batch, _, h_range, w_range = x.size()
        occluded_size = self.patch_size + 2

        input = torch.zeros(0)
        for h in range(h_range - occluded_size + 1):
            for w in range(w_range - occluded_size + 1):
                mask = torch.zeros(x.size())
                mask[:, :, h:h+occluded_size, w:w+occluded_size] = 1
                if torch.cuda.is_available(): t = x*mask.cuda()
                else:                         t = x*mask
                if input.numel() == 0: input = t
                else:                  input = torch.cat((input, t), 0)

        t = input
        for layer in self.features:
            t = layer(t)
        t = t.view(t.size(0), -1)
        out = self.classifier(t)
        return out

    def get_base_tensors(self, x):
        inter_tensors= [x]
        x = self.conv1(x)
        inter_tensors.append(x)
        x = self.conv2(x)
        inter_tensors.append(x)
        x = self.pool2(x)
        inter_tensors.append(x)
        x = self.conv3(x)
        inter_tensors.append(x)
        x = self.conv4(x)
        inter_tensors.append(x)
        x = self.pool4(x)
        inter_tensors.append(x)
        x = self.conv5(x)
        inter_tensors.append(x)
        x = self.conv6(x)
        inter_tensors.append(x)
        x = self.conv7(x)
        inter_tensors.append(x)
        x = self.pool7(x)
        inter_tensors.append(x)
        x = self.conv8(x)
        inter_tensors.append(x)
        x = self.conv9(x)
        inter_tensors.append(x)
        x = self.conv10(x)
        inter_tensors.append(x)
        x = self.pool10(x)
        inter_tensors.append(x)
        x = self.conv11(x)
        inter_tensors.append(x)
        x = self.conv12(x)
        inter_tensors.append(x)
        x = self.conv13(x)
        inter_tensors.append(x)
        x = self.pool13(x)
        inter_tensors.append(x)

        x = x.view(x.size(0), -1)
        # print('fc in shape: ', x.size())
        x = self.classifier(x)
        inter_tensors.append(x)

        return inter_tensors

    def infer_range_info(self, info, layer, h_size, w_size):
        op_list = list(filter(lambda x : type(x) == nn.Conv2d or type(x) == nn.MaxPool2d or type(x) == nn.AvgPool2d, layer.modules()))
        assert len(op_list) == 1

        op = op_list[0]

        ph = pw = sh = sw = kh = kw = 0
        if type(op.padding) == int: ph = pw = op.padding
        else:                       ph,  pw = op.padding
        if type(op.stride)  == int: sh = sw = op.stride
        else:                       sh,  sw = op.stride
        if type(op.kernel_size) == int: kh = kw = op.kernel_size
        else:                           kh,  kw = op.kernel_size

        padding = op.padding
        stride  = op.stride
        kernel  = op.kernel_size

        out_info = dict()
        out_info['hp'] = max(int((info['hp'] + ph - kh)/sh) + 1, 0)
        out_info['wp'] = max(int((info['wp'] + pw - kw)/sw) + 1, 0)
        out_info['hs'] = int((info['hp'] + info['hs'] + ph - 1)/sh) - out_info['hp'] + 1
        out_info['ws'] = int((info['wp'] + info['ws'] + pw - 1)/sw) - out_info['wp'] + 1
        out_info['hs'] = min(out_info['hs'], h_size - out_info['hp'])
        out_info['ws'] = min(out_info['ws'], w_size - out_info['wp'])

        in_info = dict()
        in_info['hp'] = out_info['hp']*sh - ph
        in_info['wp'] = out_info['wp']*sw - pw
        in_info['hs'] = (out_info['hs'] - 1)*sh + kh
        in_info['ws'] = (out_info['ws'] - 1)*sw + kw

        return in_info, out_info

    def crop_tensor(self, x, info):
        _, _, h, w = x.size()
        assert len(x.size()) == 4
        top    = max(info['hp'], 0)
        bottom = min(info['hp'] + info['hs'], h)
        left   = max(info['wp'], 0)
        right  = min(info['wp'] + info['ws'], w)
        y = x[:, :, top:bottom, left:right]

        pad = dict()
        pad['top']    = max(0, -info['hp'])
        pad['bottom'] = max(info['hp'] + info['hs'] - h, 0)
        pad['left']   = max(0, -info['wp'])
        pad['right']  = max(info['wp'] + info['ws'] - w, 0)

        return y, pad

    def run_layer(self, x, pad, layer):
        if pad['left'] != 0 or pad['right'] != 0 or pad['top'] != 0 or pad['bottom'] != 0:
            x = nn.ZeroPad2d(padding=(pad['left'], pad['right'], pad['top'], pad['bottom']))(x)
        op_list = list(filter(lambda x : type(x) == nn.Conv2d or type(x) == nn.MaxPool2d or type(x) == nn.AvgPool2d, layer.modules()))
        assert len(op_list) == 1
        op = op_list[0]
        padding = op.padding
        op.padding = [0, 0]
        x = layer(x)
        op.padding = padding

        return x

    def merge_tensor(self, part, base, info):
        assert len(part.size()) == 4
        assert len(base.size()) == 4
        base_copy = base.clone()

        base_copy[:, :, info['hp']:info['hp']+info['hs'], info['wp']:info['wp']+info['ws']] = part
        return base_copy

    def detect(self, x):
        base = self.get_base_tensors(x)
        batch, _, h_range, w_range = x.size()
        assert batch == 1

        occluded_size = self.patch_size + 2
        h_range = h_range - occluded_size + 1
        w_range = w_range - occluded_size + 1
        # print('patch size: ', self.patch_size)
        # print('occluded size: ', occluded_size)
        prob_grid = np.zeros([h_range, w_range])
        class_grid = np.zeros([h_range, w_range])
        for h in range(h_range):
            for w in range(w_range):
                # print('+++++++++++++++++++++++++++++++++++++++++++++++++')
                # print('h: {}/{}, w: {}/{}'.format(h, h_range, w, w_range))

                info = dict({'hp':h, 'wp':w, 'hs':occluded_size, 'ws':occluded_size})
                mask = torch.zeros(x.size())
                mask[:, :, h:h+occluded_size, w:w+occluded_size] = 1
                if torch.cuda.is_available(): intermediate_tensor = x*mask.cuda()
                else:                         intermediate_tensor = x*mask
                for idx, layer in enumerate(self.features):
                    # print('\n\n\nlayer id: ', idx, ' ', layer)
                    h_size = base[idx+1].size(2)
                    w_size = base[idx+1].size(3)
                    input_info, output_info = self.infer_range_info(info, layer, h_size, w_size)
                    # print('base in shape :', base[idx].size())
                    # print('info          : ', info)
                    # print('infer in info : ', input_info)
                    # print('infer out info: ', output_info)
                    input, pad = self.crop_tensor(intermediate_tensor, input_info)
                    # print('pad           : ', pad)
                    # print('input shape   : ', input.size())
                    part_out = self.run_layer(input, pad, layer)
                    # print('out shape     : ', part_out.size()) # 8 7
                    # print('base shape    : ', base[idx+1].size())
                    intermediate_tensor = self.merge_tensor(part_out, base[idx+1], output_info)
                    info = output_info
                intermediate_tensor = intermediate_tensor.view(intermediate_tensor.size(0), -1)
                intermediate_tensor = self.classifier(intermediate_tensor)
                prob, cls = torch.max(intermediate_tensor, 1)
                prob_grid[h, w] = prob
                class_grid[h, w] = cls

        return prob_grid, class_grid


def mdr_vote(prob_grid, class_grid, file_idx, prefix, logger=None):
    assert len(prob_grid.shape) == 2 and prob_grid.shape == class_grid.shape \
        and prob_grid.shape[0] > 2 and prob_grid.shape[1] > 2

    h, w = prob_grid.shape
    voting_grid = -1*np.ones([h-2, w-2])
    for i in range(h-2):
        for j in range(w-2):
            tprob = prob_grid[i:i+3, j:j+3]
            tcls  = class_grid[i:i+3, j:j+3]
            idx = np.argmin(tprob)

            voting_grid[i, j] = tcls[idx//3, idx%3]
            for k in range(9):
                if tcls[k//3, k%3] != voting_grid[i, j]:
                    voting_grid[i, j] = -1
                    break
                else:
                    continue

    freq = dict()
    for i in range(h-2):
        for j in range(w-2):
            val = int(voting_grid[i, j])
            if val == -1: continue
            else:
                if val in freq: freq[val] += 1
                else:           freq[val]  = 1
    if logger:
        logger.info('freq: {}'.format(sorted(freq.items(), key = lambda kv:kv[1], reverse=True)))
    else:
        print('freq: ', sorted(freq.items(), key = lambda kv:kv[1], reverse=True))
    if len(freq.keys()) == 1: return list(freq.keys())[0], voting_grid
    elif len(freq.keys()) == 2:
        keys = list(freq.keys())
        if freq[keys[0]] > freq[keys[1]]: return keys[1], voting_grid
        else:                            return keys[0], voting_grid
    return -1, voting_grid

def mdr_test():
    batch_size = 1
    patch_ratio = 0.1
    test_case_num = 10
    img_size = 32
    patch_size = int(math.sqrt(img_size*img_size*patch_ratio))

    print('img size: ', img_size)
    print('patch size: ', patch_size)
    print('number to run: ', (img_size - patch_size -1)*(img_size - patch_size -1))

    test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    start = time.perf_counter()
    # print('data set size: ', sum(1 for _ in test_loader))
    # exit(0)
    use_gpu = torch.cuda.is_available()
    assert img_size%32 == 0
    model = VGG16(patch_size, fc_in=512*int(img_size/32 * img_size/32))
    if use_gpu: model = model.cuda()
    for i, (img, _) in enumerate(test_loader, 1):
        if use_gpu:
            img = img.cuda()
        img = nn.functional.interpolate(img, size=[img_size, img_size], mode='bilinear')
        prob_grid, class_grid = model.detect(img)
        cls = mdr_vote(prob_grid, class_grid)
        print('predicted type: ', cls)
        if i == test_case_num: break
    end = time.perf_counter()
    print('total time: {}s, average time: {}s'.format(end-start, (end-start)/test_case_num))


if __name__ == '__main__':
    mdr_test()
    exit(0)
    batch_size = 1
    learning_rate = 1e-2
    num_epoches = 10

    train_dataset = datasets.CIFAR10('./data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    patch_size = 5
    model = VGG16(patch_size)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoches):
        print('*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader, 1):
            img, label = data

            # cuda
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            img = Variable(img)

            out = model.detect(img)
            print(out)
            exit(0)
            label = Variable(label.repeat((img.size(2)-patch_size-1)*(img.size(3)-patch_size-1)))
            # print(label.size())

            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            accuracy = (pred == label).float().mean()
            running_acc += num_correct.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))
            ))

            model.eval()
            eval_loss = 0
            eval_acc = 0
            for data in test_loader:
                img, label = data
                if use_gpu:
                    img = Variable(img, volatile=True).cuda()
                    # label = Variable(label, volatile=True).cuda()
                    label = Variable(label.repeat((img.size(2)-patch_size-1)*(img.size(3)-patch_size-1)), volatile=True).cuda()
                else:
                    img = Variable(img, volatile=True)
                    # label = Variable(label, volatile=True)
                    label = Variable(label.repeat((img.size(2)-patch_size-1)*(img.size(3)-patch_size-1)), volatile=True)
                out = model(img)
                loss = criterion(out, label)
                eval_loss += loss.item() * label.size(0)
                _, pred = torch.max(out, 1)
                num_correct = (pred == label).sum()
                eval_acc += num_correct.item()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            test_dataset
        )), eval_acc / (len(test_dataset))))
        print()

    torch.save(model.state_dict(), './vgg16.pth')
