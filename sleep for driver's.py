import cv2
import platform

# Try importing sound modules
try:
    import winsound
    beep_func = lambda: winsound.Beep(1000, 1000)
except ImportError:
    try:
        from playsound import playsound
        beep_func = lambda: playsound("alarm.wav")  # You must have alarm.wav in same folder
    except:
        beep_func = lambda: print("[!] No sound support available.")

# Load classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

cap = cv2.VideoCapture(0)

closed_eyes_frames = 0
yawn_frames = 0
sleep_threshold = 30
yawn_threshold = 15
alert_triggered = False

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    sleepy = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(30, 30))
        if len(eyes) == 0:
            closed_eyes_frames += 1
        else:
            closed_eyes_frames = 0

        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.7, 20)
        yawning = False
        for (mx, my, mw, mh) in mouth:
            if my > h / 2:
                cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)
                yawning = True
                break

        if yawning:
            yawn_frames += 1
        else:
            yawn_frames = 0

        if closed_eyes_frames > sleep_threshold or yawn_frames > yawn_threshold:
            sleepy = True
        else:
            alert_triggered = False

    if sleepy:
        cv2.putText(frame, "DROWSINESS ALERT!", (50, 50), font, 1, (0, 0, 255), 3)
        if not alert_triggered:
            print("[!] Beep alert triggered!")
            beep_func()
            alert_triggered = True
    else:
        cv2.putText(frame, "Status: Awake", (50, 50), font, 1, (0, 255, 0), 2)

    cv2.imshow("Sleepiness Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
