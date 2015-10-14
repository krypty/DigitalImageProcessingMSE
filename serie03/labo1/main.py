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
def mean_filter(img, size=3):
    img_meaned = np.copy(img)

    ker_mean = np.ones([size, size]) / size**2
    # manual method for a kernel of size 3
    # ker_mean = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0
    return cv2.filter2D(img_meaned, ddepth=cv2.CV_8U, kernel=ker_mean)


@OpenCVUtils.timeit
def mean_filter_ocv(img, size=3):
    return cv2.blur(img, (size, size))


@OpenCVUtils.timeit
def gaussian_filter(img, sigma, size=3):

    def g(x, y):
        return (1.0/(2.0*math.pi*(sigma**2)))*math.exp(-((x**2+y**2)/(2.0*sigma**2)))

    img_gaussian = np.copy(img)
    ker_gaussian = np.zeros([size, size])

    sum_ker = 0
    for i in range(size):
        for j in range(size):
            v = g(i, j)
            ker_gaussian[i][j] = v
            sum_ker += v

    ker_gaussian /= sum_ker

    return cv2.filter2D(img_gaussian, ddepth=cv2.CV_8U, kernel=ker_gaussian)


@OpenCVUtils.timeit
def gaussian_filter_ocv(img, sigma, size=3):
    ker = cv2.getGaussianKernel(ksize=size, sigma=sigma)
    return cv2.filter2D(img, ddepth=cv2.CV_8U, kernel=ker)

if __name__ == '__main__':

    # load and display original lena
    img_lena = cv2.imread(r'./img/lena.pgm', cv2.IMREAD_GRAYSCALE)
    cv2.imshow("lena", img_lena)

    # mean filter
    img_lena_meaned = mean_filter(img_lena, size=5)
    img_lena_meaned_ocv = mean_filter_ocv(img_lena, size=5)
    cv2.imshow("lena meaned", img_lena_meaned)
    cv2.imshow("lena meaned ocv", img_lena_meaned_ocv)

    # gaussian blur
    img_lena_gaussian = gaussian_filter(img_lena, size=7, sigma=0.84089642)
    img_lena_gaussian_ocv = gaussian_filter_ocv(img_lena, sigma=0.84089642, size=7)
    cv2.imshow("lena gaussian", img_lena_gaussian)
    cv2.imshow("lena gaussian ocv", img_lena_gaussian_ocv)

    cv2.waitKey()
    cv2.destroyAllWindows()
