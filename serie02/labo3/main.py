#!/usr/bin/python
# coding: utf8

import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

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
def histogram_equalisation_ocv(img):
    return cv2.equalizeHist(img)


@OpenCVUtils.timeit
def get_histogram(img):
    _hist = np.zeros(255)

    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            color = img[i][j]
            _hist[color] += 1

    return _hist


@OpenCVUtils.timeit
def show_histograms():
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    hist = get_histogram(img_lena)
    ax1.plot(hist, color="red")
    hist_original, bins = np.histogram(img_lena.ravel(), 256, [0, 256])
    ax2.plot(hist_original, color="blue")
    plt.show()


@OpenCVUtils.timeit
def get_image_average(img):
    pixels = img.flatten()
    return sum(pixels) / len(pixels)


@OpenCVUtils.timeit
def get_contrast(img):
    pixels = img.flatten()
    return max(pixels) - min(pixels)


@OpenCVUtils.timeit
def get_entropy(img):
    histogram = get_histogram(img)
    sum_histogram = sum(histogram)

    entropy = 0
    for l in filter(lambda x: x > 0, histogram):
        p = l/sum_histogram
        entropy += p * math.log2(p)

    return -entropy


if __name__ == '__main__':
    img_lena = cv2.imread(r'./img/LenaX.png', cv2.IMREAD_GRAYSCALE)

    # histogram
    show_histograms()

    # average
    color_avg = get_image_average(img_lena)
    print("average = %s" % color_avg)

    # contrast
    contrast = get_contrast(img_lena)
    print("contrast: %s" % contrast)

    # entropy
    entropy = get_entropy(img_lena)
    print(entropy)

    cv2.waitKey()
    cv2.destroyAllWindows()
