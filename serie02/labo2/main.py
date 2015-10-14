#!/usr/bin/python
# coding: utf8

import cv2

if __name__ == '__main__':
    # dollar100 = r'./img/100dollarsA.tif'
    # dollar100 = r'./img/100dollarsB.tif'
    dollar100 = r'./img/100dollarsC.tif'
    # dollar100 = r'./img/LenaX.png'

    imgDollar100 = cv2.imread(dollar100)

    imgDollar100 = cv2.cvtColor(imgDollar100, cv2.COLOR_BGR2GRAY)
    cv2.imshow("dollar", imgDollar100)

    mask = 0x01
    for b in range(1, 9):
        print("{0:b}".format(mask).zfill(8))
        imgDollar100 = cv2.bitwise_and(imgDollar100, mask)
        imgDollar100 = cv2.normalize(imgDollar100, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow("mask %s" % mask, imgDollar100)

        mask <<= 1

    cv2.waitKey()
    cv2.destroyAllWindows()
