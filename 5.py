import cv2
img= cv2.imread("nature.png",0)

ret,thresh_img = cv2.threshold(img , 0 ,255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print ("Optimal threshold value:",ret)
