#!/usr/bin/python
# coding: utf8

import cv2
from matplotlib import pyplot as plt
import numpy as np

# this import OpenCVUtils
import os
import sys
filename = os.path.abspath("../../OpenCVUtils.py")

directory, module_name = os.path.split(filename)
module_name = os.path.splitext(module_name)[0]

path = list(sys.path)
sys.path.insert(0, directory)
try:
    OpenCVUtils = __import__(module_name)
finally:
    sys.path[:] = path  # restore


@OpenCVUtils.timeit
def negative(img):
    img_negative = np.copy(img)

    rows, cols = img_negative.shape
    for i in range(rows):
        for j in range(cols):
            img_negative[i][j] = 255 - img_negative[i][j]

    return img_negative


@OpenCVUtils.timeit
def gamma_correction(img, gamma):
    img_negative = np.copy(img)

    rows, cols = img_negative.shape
    for i in range(rows):
        for j in range(cols):
            img_negative[i][j] = 255 * (img[i][j] / 255)**gamma

    return img_negative


@OpenCVUtils.timeit
def contrast_stretching(img):
    img_result = np.copy(img)

    s_max = 255
    r_min = img_result.min()
    r_max = img_result.max()

    def adjust(r):
        return s_max * ((r - r_min)/(r_max - r_min))

    rows, cols = img_result.shape
    for i in range(rows):
        for j in range(cols):
            img_result[i][j] = adjust(img[i][j])

    return img_result


@OpenCVUtils.timeit
def histogram_equalisation_manual(img):
    img_result = np.copy(img)

    L = 256
    rows, cols = img_result.shape
    MN = rows*cols

    # compute histogram -> h(f)
    h, bins = np.histogram(img_result.ravel(), L, [0, L])

    # OMG that part will break your heart...and make you love Python
    # see formula (3.3.8) in Gonzales Woods
    # we compute the normalized cdf from the histogram and we will use it as a lookup table
    s_k = [int(((L-1)/MN)*sum(h[:k]) + 0.5) for k in range(L-1)]

    for i in range(rows):
        for j in range(cols):
            # here we use it as a lookup table
            img_result[i][j] = s_k[(img[i][j])]

    return img_result


@OpenCVUtils.timeit
def histogram_equalisation_ocv(img):
    img_result = cv2.equalizeHist(img)
    return img_result


if __name__ == '__main__':

    bad_photo = r'./img/MauvaisePhoto.tif'
    mire_originale = r'./img/mireOriginale.tif'
    mire_degradee = r'./img/mireDegradee.tif'

    img_bad_photo = cv2.imread(bad_photo, cv2.IMREAD_GRAYSCALE)
    img_mire_originale = cv2.imread(mire_originale, cv2.IMREAD_GRAYSCALE)
    img_mire_degradee = cv2.imread(mire_degradee, cv2.IMREAD_GRAYSCALE)

    # negative
    img_bad_photo_negative = negative(img_bad_photo)
    cv2.imshow("img_bad_photo_negative", img_bad_photo_negative)

    # gamma correction
    img_mire_fixed = gamma_correction(img_mire_degradee, 0.4)
    cv2.imshow("img_mire_fixed (gamma correction)", img_mire_fixed)
    cv2.imshow("img_mire_originale", img_mire_originale)

    # histogram equalisation
    img_bad_photo_stretched = contrast_stretching(img_bad_photo)
    img_bad_photo_equalized_ocv = histogram_equalisation_ocv(img_bad_photo)
    cv2.imshow("img_bad_photo", img_bad_photo)
    cv2.imshow("img_bad_photo_stretched", img_bad_photo_stretched)
    cv2.imshow("img_bad_photo_equalized_ocv", img_bad_photo_equalized_ocv)

    img_bad_photo_equalized_manual = histogram_equalisation_manual(img_bad_photo)
    cv2.imshow("img_bad_photo_equalized_manual", img_bad_photo_equalized_manual)

    cv2.waitKey()
    cv2.destroyAllWindows()
