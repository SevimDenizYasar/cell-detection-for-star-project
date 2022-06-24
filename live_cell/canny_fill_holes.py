# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 22:28:27 2022

@author: sdeni
"""

import numpy as np
import argparse
import cv2

def fill_holes(imInput, threshold):
    # Threshold.
    th, thImg = cv2.threshold(imInput, threshold, 255, cv2.THRESH_BINARY_INV)

    imFloodfill = thImg.copy()

    h, w = thImg.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(imFloodfill, mask, (0,0), 255)

    imFloodfillInv = cv2.bitwise_not(imFloodfill)

    imOut = thImg | imFloodfillInv

    return imOut

if __name__ == "__main__":
    image = cv2.imread("../input_images/holes.jpeg")
    cv2.imshow("Original image", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    #cv2.imshow("Blurred", blurred)

    # Fille the "holes" on the image.
    filled = fill_holes(blurred, 200)
    cv2.imshow("Filled", filled)
    ret, filled_1 = cv2.threshold(filled, 0, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Filled1", filled_1)

    # Find circles by the Hough transfermation.
    circles = cv2.HoughCircles(filled_1, cv2.HOUGH_GRADIENT, 2, 700, param1 = 255, param2 = 15, minRadius = 0, maxRadius = 20)
    
   
    count = 0
    if circles is not None:
        circles = np.uint8(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(filled_1, (i[0], i[1]), i[2], (0, 255, 0), 2)
            count += 1

    cv2.imshow("Filled1", filled_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()