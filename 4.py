import cv2

img=cv2.imread("nature.png",cv2.IMREAD_GRAYSCALE)
_, th_binary = cv2.threshold (img, 124 , 255 , cv2.THRESH_BINARY)
_, th_binary_inv = cv2.threshold (img, 124 , 255 ,cv2.THRESH_BINARY_INV)
_, th_trunc = cv2.threshold  (img, 124 , 255 ,cv2.THRESH_TRUNC)
_, th_tozero = cv2.threshold  (img, 124 , 255 ,cv2.THRESH_TOZERO)
_, th_tozero_inv = cv2.threshold  (img, 124 , 255 ,cv2.THRESH_TOZERO_INV)

cv2.imshow("Original", img)
cv2.imshow("Binary", th_binary)
cv2.imshow("Binary Inverse", th_binary_inv)
cv2.imshow("Trunc", th_trunc)
cv2.imshow("To Zero", th_tozero)
cv2.imshow("To Zero Inverse", th_tozero_inv)

cv2.waitkey(0)
cv2.destoryAllWidows()

