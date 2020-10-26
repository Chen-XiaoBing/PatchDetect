# https://blog.csdn.net/qq_27261889/article/details/90675051
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import fftpack

def high_pass_filter(img, radius=80):
    r = radius

    rows, cols = img.shape
    center = int(rows / 2), int(cols / 2)

    mask = np.ones((rows, cols, 2), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0
    return mask

def low_pass_filter(img, radius=100):
    r = radius

    rows, cols = img.shape
    center = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols, 2), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 1
    return mask

def bandreject_filters(img, r_out=300, r_in=35):
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    radius_out = r_out
    radius_in = r_in

    mask = np.zeros((rows, cols, 2), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                               ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
    mask[mask_area] = 1
    mask = 1 - mask
    return mask

def guais_low_pass(img, radius=10):
    rows, cols = img.shape
    center = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols, 2), np.float32)
    x, y = np.ogrid[:rows, :cols]
    for i in range(rows):
        for j in range(cols):
            distance_u_v = (i - center[0]) ** 2 + (j - center[1]) ** 2
            mask[i, j] = np.exp(-0.5 *  distance_u_v / (radius ** 2))
    return mask


def guais_high_pass(img, radius=10):
    rows, cols = img.shape
    center = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols, 2), np.float32)
    x, y = np.ogrid[:rows, :cols]
    for i in range(rows):
        for j in range(cols):
            distance_u_v = (i - center[0]) ** 2 + (j - center[1]) ** 2
            mask[i, j] = 1 - np.exp(-0.5 *  distance_u_v / (radius ** 2))
    return mask


def laplacian_filter(img, radius=10):
    rows, cols = img.shape
    center = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols, 2), np.float32)
    x, y = np.ogrid[:rows, :cols]
    for i in range(rows):
        for j in range(cols):
            distance_u_v = (i - center[0]) ** 2 + (j - center[1]) ** 2
            mask[i, j] = -4 * np.pi ** 2 * distance_u_v
    return mask

def log_magnitude(img):
    magnitude_spectrum = 20 * np.log(cv2.magnitude(img[:, :, 0], img[:, :, 1]))
    return magnitude_spectrum

def get_condidate(fimg, thre=200):
    img = cv2.imread(fimg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # first step: compute the FFT of original images
    img_dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    img_dft_shift = np.fft.fftshift(img_dft)

    # second step: compute the mask
    # mask = bandreject_filters(img, r_out=90, r_in=40)
    # mask = high_pass_filter(img, radius=50)
    # mask = low_pass_filter(img, radius=100)
    # mask = guais_low_pass(img, radius=30)
    # mask = guais_high_pass(img, radius=50)
    mask = laplacian_filter(img, radius=50)

    # third step: fft of original images multiply the filter
    fshift = img_dft_shift * mask

    # do log to minize the region
    spectrum_after_filtering = log_magnitude(fshift)

    # Fourth step: IFFT
    f_ishift = np.fft.ifftshift(fshift)
    img_after_filtering = cv2.idft(f_ishift)
    img_after_filtering = log_magnitude(img_after_filtering)

    tmp = list(img_after_filtering.flatten())
    tmp.sort(reverse=True)
    thre = tmp[thre]
    img_after_filtering = (img_after_filtering >= thre).astype(int)

    return img_after_filtering

# get_condidate("data/attack_pic_2/patched_resnet50_0.1_50_37.jpg")
