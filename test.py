#!/usr/bin/python  
# -*- coding: <encoding name> -*-  

import cv2 as cv

img = cv.imread("./img/6.jpg")
cv.namedWindow("Image")
cv.imshow("Image",img)
cv.waitKey(0)
cv.destroyAllWindows()