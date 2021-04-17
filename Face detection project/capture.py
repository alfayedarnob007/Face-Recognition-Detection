import cv2 ,time 
#object creation
video = cv2.VideoCapture(0)
a= 0

while True:
    a = a + 1

    #3 frame object
    check , frame = video.read()
    print(check)
    print(frame)
    #gray scale convertion
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #4 show the frames
    cv2.imshow('Capturing',gray)
    #5 key press to exit
    #cv2.waitKey(0)
    #for video playing
    key = cv2.waitKey(1)

    if key == ord('q'):
        break 
print(a)
#2shutdown cam
video.release()

cv2.destroyAllWindows