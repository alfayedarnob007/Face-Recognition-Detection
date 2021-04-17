import cv2 ,time
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
#object creation
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
colour1 = (255,0,0)
colour2 = (0,255,0)
colour3 = (0,0,255)
colour4 = (255,0,55)
colour = [colour1,colour2,colour3,colour4]



while (True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces= face_cascade.detectMultiScale (gray, 1.5, 5)
	for i , (x,y,w,h) in enumerate(faces) :
		print(x,y,w,h)
		roi_gray = gray[y:y+h,x:x+w]
		roi_color = frame[y:y+h,x:x+w]
		image = 'arnob_image.png'
		cv2.imwrite(image,roi_gray)
		number_of_faces = str(len(faces))
		
			
		cv2.rectangle(frame,(x,y),(x+w,y+h),colour[i],2)
			

		
		cv2.putText(frame,'Number of Faces : ' + number_of_faces,(40, 40),cv2.FONT_HERSHEY_SIMPLEX , 1,(255,0,0),2) 

	cv2.imshow('frame',frame)
	key = cv2.waitKey(20)  
	if key == ord('q'):
    		break
		

cap.release()
cv2.destroyAllWindows