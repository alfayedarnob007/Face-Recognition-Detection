import cv2 ,time
import numpy as np
import argparse
import os
import imutils
import keras
import pickle
from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model


# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
# 	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to Caffe pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
# 	help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())

colour1 = (155,50,50)
colour2 = (0,255,0)
colour3 = (0,0,255)
colour4 = (155,50,50)
colour5 = (50,50,155)

colour = [colour1,colour2,colour3,colour4,colour5]

protoType = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoType, model)



vs = VideoStream(src=0).start()
time.sleep(2.0)
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)


while True :
        frame = vs.read()
        frame = imutils.resize(frame,1000)
        (h,w) = frame.shape[:2]
        resized = cv2.resize(frame,(300,300))
        blob = cv2.dnn.blobFromImage(resized,1.0, (300, 300), (104, 117, 123))

        net.setInput(blob)
        detections = net.forward()


        for i in range(0,detections.shape[2]):

                confidence = detections[0, 0, i, 2]

                if confidence > 0.5 :
                        box = detections[0,0,i,3:7]* np.array([w,h,w,h])
                        (startX, startY, endX, endY) = box.astype("int")

                        

			# ensure the detected bounding box does fall outside the
			# dimensions of the frame
                        startX= max(0,startX)
                        startY = max(0,startY)
                        endX = min(w,endX)
                        endY = min(h,endY)
                        face = frame[startY:endY,startX:endX]
                        face = cv2.resize(frame,(32,32))
                        face = face.astype('float')/255
                        face = img_to_array(face)
                        face = np.expand_dims(face,axis=0)
		        
                        # preds = model.predict(face)[0]
                        # j = np.argmax(preds)
                        # label = le.classes_[j]

                        # label = '{}: {:.4f}'.format(label,preds[j])
                        number_of_faces = str(len(face))
                        cv2.putText(frame,'Number of Faces : ' + number_of_faces,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX , 0.5,(255,0,0),2)
                        #cv2.putText(frame,'Face Detected',(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                        cv2.rectangle(frame,(startX,startY),(endX,endY),colour[i],2)


        cv2.imshow('Frame',frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
                break

	
cv2.destroyAllWindows()
vs.stop()
       

#while (True):
    # 	ret, frame = cap.read()
# 	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# 	faces= face_cascade.detectMultiScale (gray, 1.5, 5)
# 	for i , (x,y,w,h) in enumerate(faces) :
# 		print(x,y,w,h)
# 		roi_gray = gray[y:y+h,x:x+w]
# 		roi_color = frame[y:y+h,x:x+w]
# 		image = 'arnob_image.png'
# 		cv2.imwrite(image,roi_gray)
# 		number_of_faces = str(len(faces))
		
			
# 		cv2.rectangle(frame,(x,y),(x+w,y+h),colour[i],2)
			

		
# 		cv2.putText(frame,'Number of Faces : ' + number_of_faces,(40, 40),cv2.FONT_HERSHEY_SIMPLEX , 1,(255,0,0),2) 
