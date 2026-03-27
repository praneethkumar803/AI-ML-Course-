import cv2
img1=cv2.imread("nature.png")
cv2.imshow('show',img1)
cv2.imwrite('photo.jpeg',img1)
cv2.waitKey(5000)
cv2.destroyAllWindows()
