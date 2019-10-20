import numpy as np
import cv2
import easygui




face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Opening an image from a file
f = easygui.fileopenbox()
img = cv2.imread(f)


#define the screen resulation
screen_res = 1280, 720
# scale_width = screen_res[0] / img.shape[1]
# scale_height = screen_res[1] / img.shape[0]
# scale = min(scale_width, scale_height)
#resized window width and height
# window_width = int(img.shape[1])/3
# window_height = int(img.shape[0])
 
#cv2.WINDOW_NORMAL makes the output window resizealbe
cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
 
#resize the window according to the screen resolution
# cv2.resizeWindow('Resized Window', window_width, window_height)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
(x,y,w,h) = faces[0]
photo_crop = img[y-35:y+h+80,x-40:x+w+50]
# img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
roi_gray = gray[y:y+h, x:x+w]
roi_color = img[y:y+h, x:x+w]
    
# sign = cv2.rectangle(img,(x-w-170,y+h+160),(x+w+50,y+h+320),(255,0,0),2)
sign_crop = img[y+h+160:y+h+320,x-w-170:x+w+50]

cv2.imshow('Resized Window',img)
cv2.imshow("Cropped Photo", photo_crop)
cv2.imshow("Cropped Sign", sign_crop)
cv2.imwrite('cropped_photo.jpg',photo_crop)
cv2.imwrite('cropped_sign.jpg',sign_crop)

cv2.waitKey(0)
cv2.destroyAllWindows()