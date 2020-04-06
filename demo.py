import cv2,os
# cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))

# haar_model = os.path.join(cv2_base_dir, 'haarcascade_frontalface_default.xml')
# print(haar_model)
face_cascade=cv2.CascadeClassifier('C:\\Users\\Saloni Darji\\Desktop\\face recognition\\haarcascade_frontalface_default.xml')
print("face cascade: ", face_cascade)
img = cv2.imread("images/papa.jpg",1)
print(img.shape)
# resize= cv2.resize(img,(int(img.shape[1]/2), int(img.shape[0]/2)))

gray_img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print("gray: ",gray_img)
faces= face_cascade.detectMultiScale(gray_img,scaleFactor=1.05, minNeighbors=5)
print(faces)
print(type(faces))


for x,y,w,h in faces:
    img= cv2.rectangle(img, (x,y) , (x+w,y+h),(0,255,0),3)
    
cv2.imshow("my image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


