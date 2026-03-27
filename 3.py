import cv2
import imutils
import sys

Img =cv2.imread("nature.png")

resizedIng =imutils.resize(Img, width=500)

cv2.imshow("OriginalImg2.png",Img)

cv2.imshow('resized.png',resizedIng)

cv2.imwrite('resizedImg2.png', resizedIng)

cv2.waitKey(0)

cv2.destroyAllWindows()
