import cv2,time, pandas
from datetime import datetime

face_cascade=cv2.CascadeClassifier('C:\\Users\\Saloni Darji\\Desktop\\face recognition\\haarcascade_frontalface_default.xml')
first_frame=None
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["Start","End"])

video=cv2.VideoCapture(0)
a=1
while True:
    a=a+1
    check,frame=video.read()
    status=0
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (21,21), 0)    
    
    faces= face_cascade.detectMultiScale(gray,scaleFactor=1.05, minNeighbors=5)
    print(faces)
    print(type(faces))
    
    
    for x,y,w,h in faces:
        frame= cv2.rectangle(frame, (x,y) , (x+w,y+h),(0,255,0),3)
    
    cv2.imshow("Capture", frame)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
    
    if first_frame is None:
        first_frame=gray
        continue

print(a)
video.release()
cv2.destroyAllWindows() 