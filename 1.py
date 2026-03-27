import cv2
import sys

image = cv2.imread("nature.png")

grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

cv2.imshow('original', image)

cv2.imshow('Gray', grayImage)

cv2.imwrite('graynew.jpeg', grayImage)

cv2.waitKey(10000)

cv2.destroyAllWindows()
