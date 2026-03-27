#Program for the profileface
import cv2
profile_cascade = cv2.CascadeClassifier("haarcascade_profileface.xml")


cam = cv2.VideoCapture(0)

while True:
    
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  
    faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

  
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3) 

 
    cv2.imshow("Side Face Detection (Press ESC to exit)", img)

    
    key = cv2.waitKey(1)
    if key == 27:
        break


cam.release()
cv2.destroyAllWindows()
