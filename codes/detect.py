import os
import pdb
import logging
import numpy as np

import torch
import torch.nn.functional as F

from freq_filter import get_condidate
from mylogger import get_mylogger

class Detector:
    def __init__(self, file_dir, logger):
        self.file_dir = file_dir
        self.logger = logger
        self.result = list()
        files = os.listdir(self.file_dir)

        self.file_dict = dict()
        ori_img, pat_img, ori_mdr, pat_mdr = dict(), dict(), dict(), dict()
        for item in files:
            suffix_jpg     = item.endswith('.jpg')
            suffix_mdr_pth = item.endswith('_mdr.pth')
            prefix_ori     = item.startswith('ori_')
            prefix_pat     = item.startswith('patched_')

            if suffix_mdr_pth and prefix_ori:
                key = int(item.split('_')[3])
                ori_mdr[key] = item
            elif suffix_mdr_pth and prefix_pat:
                key = int(item.split('_')[3])
                pat_mdr[key] = item
            elif suffix_jpg and prefix_ori:
                key = int(item.split('_')[3])
                ori_img[key] = item
            elif suffix_jpg and prefix_pat:
                key = int(item.split('_')[3])
                pat_img[key] = item

        self.file_dict['ori_img'] = ori_img
        self.file_dict['pat_img'] = pat_img
        self.file_dict['ori_mdr'] = ori_mdr
        self.file_dict['pat_mdr'] = pat_mdr

    def read_mdr(self, fname, key):
        assert key in ['prob', 'cls', 'vote']
        if '/' not in fname:
            fname = os.path.join(self.file_dir, fname)
        return torch.load(fname, map_location=torch.device('cpu'))[key]

    def read_img(self, fname):
        fname = os.path.join(self.file_dir, fname)
        return torch.load(fname, map_location=torch.device('cpu')).numpy()

    def get_patch_loc(self, idx):
        ori_img = self.read_img(self.file_dict['ori_img'][idx])
        pat_img = self.read_img(self.file_dict['pat_img'][idx])
        diff = ori_img - pat_img
        _, _, h, w = np.nonzero(diff)
        return h.min(), w.min(), h.max()-h.min()+1, w.max()-w.min()+1

    def detect(self, fimg, fmdr, thre, target=859, square_size=70):
        cond_mat = get_condidate(fimg, thre=thre)
        mdr_mat = self.read_mdr(fmdr, 'cls')
        sparse_mdr_mat = np.zeros(mdr_mat.shape) - 1
        # sparse_mdr_mat = 859
        h, w = cond_mat.shape
        for i in range(h):
            for j in range(w):
                if cond_mat[i, j] == 1:
                    ti = min(max(0, i-35), 154)
                    tj = min(max(0, j-35), 154)
                    sparse_mdr_mat[ti, tj] = mdr_mat[ti, tj]

        h, w = sparse_mdr_mat.shape
        pdb.set_trace()
        cond_nr  = np.zeros((h-square_size+1, w-square_size+1)).astype(int)
        other_nr = np.zeros((h-square_size+1, w-square_size+1)).astype(int)

        # for i in range(h-square_size+1):
        #     for j in range(w-square_size+1):
        #         t_cond_nr, t_other_nr = 0, 0
        #         for ii in range(square_size):
        #             for jj in range(square_size):
        #                 if sparse_mdr_mat[i+ii, j+jj] != -1:
        #                     t_cond_nr += 1
        #                     if sparse_mdr_mat[i+ii, j+jj] != 859:
        #                         t_other_nr += 1
        #         cond_nr[i, j] = t_cond_nr
        #         other_nr[i, j] = t_other_nr
        for i in range(h-square_size+1):
            for j in range(w-square_size+1):
                if i == 0 and j == 0:
                    t_cond_nr, t_other_nr = 0, 0
                    for ii in range(square_size):
                        for jj in range(square_size):
                            if sparse_mdr_mat[i+ii, j+jj] != -1:
                                t_cond_nr += 1
                                if sparse_mdr_mat[i+ii, j+jj] != target:
                                    t_other_nr += 1
                    f_cond_nr, f_other_nr = t_cond_nr, t_other_nr
                elif i != 0 and j == 0:
                    for jj in range(square_size):
                        if sparse_mdr_mat[i-1, jj] != -1:
                            f_cond_nr -= 1
                            if sparse_mdr_mat[i-1, jj] != target:
                                f_other_nr -= 1
                    for jj in range(square_size):
                        if sparse_mdr_mat[i+square_size-1, jj] != -1:
                            f_cond_nr += 1
                            if sparse_mdr_mat[i+square_size-1, jj] != target:
                                f_other_nr += 1
                    t_cond_nr, t_other_nr = f_cond_nr, f_other_nr
                else:
                    for ii in range(square_size):
                        if sparse_mdr_mat[i+ii, j+square_size-1] != -1:
                            t_cond_nr += 1
                            if sparse_mdr_mat[i+ii, j+square_size-1] != target:
                                t_other_nr += 1
                cond_nr[i, j], other_nr[i, j] = t_cond_nr, t_other_nr
                for ii in range(square_size):
                    if sparse_mdr_mat[i+ii, j] != -1:
                        t_cond_nr -= 1
                        if sparse_mdr_mat[i+ii, j] != target:
                            t_other_nr -= 1

        nr = np.count_nonzero(sparse_mdr_mat+1)

        ind = np.unravel_index(np.argsort(cond_nr, axis=None), cond_nr.shape)
        tcond_nr = cond_nr[ind[0][-1]][ind[1][-1]]
        tother_nr = other_nr[ind[0][-1]][ind[1][-1]]
        top = (tcond_nr, tother_nr, tcond_nr*1./nr, tother_nr*1./tcond_nr)
        self.logger.info('sample: {}'.format(fimg))
        self.logger.info('cond: {}, box: {}, other: {}, cond rate: {}, other label rate: {}'\
                .format(nr, top[0], top[1], top[2], top[3]))
        self.logger.info('')
        # image_file condidate_number condidate_in_box
        # other_label_in_box cond_rate other_label_rate
        self.result.append((fimg, nr, top[0], top[1], top[2], top[3]))

    def search_thre(self):
        self.logger.info('Searching threshold..')
        min_val, thre1, thre2, fpb, tnb = len(self.result), 0, 0, 0, 0
        for ii in range(100):
           for jj in range(100):
               t1, t2 = 0.01*ii, 0.01*ii
               fp, tn = 0, 0
               for i in range(len(self.result)):
                   if 'ori' in self.result[i][0] \
                           and self.result[i][-2] > t1 \
                           and self.result[i][-1] >= t2:
                       fp += 1
                   if 'patch' in self.result[i][0] \
                           and (self.result[i][-2] <= t1 \
                           or self.result[i][-1] < t2):
                       tn += 1
               if fp+tn < min_val:
                   min_val, thre1, thre2, fpb, tnb = fp+tn, 0.01*ii, 0.01*jj, fp, tn
               self.logger.info('thre1: {}, thre2: {}, fp: {}, tn: {}, total: {}'\
                       .format(0.01*ii, 0.01*jj, fp, tn, (fp+tn)/len(self.result)))

        self.logger.info('Best Result: thre1: {}, thre2: {}, fp: {}, tn: {}, total: {}'\
                .format(thre1, thre2, fpb, tnb, (fpb+tnb)/len(self.result)))


def main():
    logger = get_mylogger('log/detect_log')
    detector = Detector('./result/attack/image_specific/ResNet18/ResNet18_10/', logger)

    for i in range(10000):
        if i in detector.file_dict['ori_mdr']:
            img_file = os.path.join(detector.file_dir, detector.file_dict['ori_img'][i])
            mdr_file = os.path.join(detector.file_dir, detector.file_dict['ori_mdr'][i])
            target = int(img_file[:-4].split('_')[-1])
            detector.detect(img_file, mdr_file, 200, target=target)

        if i in detector.file_dict['pat_mdr']:
            img_file = os.path.join(detector.file_dir, detector.file_dict['pat_img'][i])
            mdr_file = os.path.join(detector.file_dir, detector.file_dict['pat_mdr'][i])
            detector.detect(img_file, mdr_file, 200, target=5)
    detector.search_thre()

if __name__ == '__main__':
    main()
