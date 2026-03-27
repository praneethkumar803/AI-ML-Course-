import cv2
img=cv2.imread("nature.png")
'''
syntax ;
dst=cv2.GassianBlur (src,(kernel),borderType)
'''
gaussianImg = cv2.GaussianBlur(img,(41,41),0)
gaussianImg1 = cv2.GaussianBlur(img,(21,21),0)
cv2.imshow("original",img)
cv2.imshow("gaussianImg",gaussianImg)
cv2.imshow("gaussianImg1",gaussianImg1)
