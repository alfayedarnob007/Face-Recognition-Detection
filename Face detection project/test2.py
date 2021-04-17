import cv2 
protoType = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(protoType, model)