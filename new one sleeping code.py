import cv2
import platform

# Sound alert setup depending on OS
try:
    import winsound
    def beep():
        winsound.Beep(1000, 1000)  # Windows beep
except ImportError:
    from playsound import playsound
    def beep():
        playsound("alarm.wav")  # Use your own alarm.wav here

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

cap = cv2.VideoCapture(0)

closed_eyes_frames = 0
yawn_frames = 0

# Thresholds (tune as needed)
EYE_CLOSED_THRESHOLD = 30   # ~1 second at 30fps
YAWN_THRESHOLD = 15         # ~0.5 seconds

alert_triggered = False

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    sleepy = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30,30))
        if len(eyes) == 0:
            closed_eyes_frames += 1
        else:
            closed_eyes_frames = 0
            alert_triggered = False  # reset alert when eyes open

        # Detect mouth (yawn)
        mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
        yawning = False
        for (mx, my, mw, mh) in mouth:
            if my > h / 2:  # mouth should be in lower half
                yawning = True
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 255, 0), 2)
                break

        if yawning:
            yawn_frames += 1
        else:
            yawn_frames = 0
            alert_triggered = False  # reset alert when not yawning

        # Check drowsiness
        if closed_eyes_frames > EYE_CLOSED_THRESHOLD or yawn_frames > YAWN_THRESHOLD:
            sleepy = True

        # Draw face and eyes rectangles for debugging
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)

    if sleepy:
        cv2.putText(frame, "DROWSINESS ALERT!", (50, 50), font, 1, (0, 0, 255), 3)
        if not alert_triggered:
            beep()  # Play beep once
            alert_triggered = True
    else:
        cv2.putText(frame, "Status: Awake", (50, 50), font, 1, (0, 255, 0), 2)

    cv2.imshow("Sleepiness Detector", frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
