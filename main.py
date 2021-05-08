import cv2
import numpy as np

clickNumber = 0

def onClick(event, x, y, flags, param):
    global clickNumber

    if(event == cv2.EVENT_LBUTTONDOWN):
        if(clickNumber == 3):
            clickNumber = 1
        else:
            clickNumber += 1
    elif(event == cv2.EVENT_RBUTTONDOWN):
        clickNumber = 0

def applyEffect(frame):
    if(clickNumber == 1):
        return cv2.blur(frame, (50, 50))
    elif(clickNumber == 2):
        canny = cv2.Canny(frame, 100, 100)
        return cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    elif(clickNumber == 3):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_OTSU)
        return cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
        
    return frame


#Definindo evento
cv2.namedWindow('video')
cv2.setMouseCallback('video', onClick)

#Definindo filtro de detecção de face
path_face = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_classifier = cv2.CascadeClassifier(path_face) 

#Capturando vídeo
videoCapture = cv2.VideoCapture(0)

if videoCapture.isOpened():
    isOpened, frame = videoCapture.read()
else:
    isOpened = False

#Realizando repetição enquanto a tecla ESC não for pressionada
while isOpened:
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    faces_return = face_classifier.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 5)
    for (x, y, w, h) in faces_return:
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        frame[y:y+h, x:x+w] = applyEffect(frame[y:y+h, x:x+w])

    cv2.imshow("video", frame)
    isOpened, frame = videoCapture.read()

    key = cv2.waitKey(5)
    if key == 27:
        break

videoCapture.release()
cv2.destroyAllWindows()