
import numpy as np
import cv2
import time


print("!! Invisibility is no more a Dream !!")
window_name='Eureka !!'
# Capturing Webcam Feed
cap = cv2.VideoCapture(0)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
time.sleep(3)
count = 0
background = 0
# Capturing Static Background Frame
for i in range(60):
    ret, background = cap.read()
# Flip the Image
background = np.flip(background, axis=1)
start=time.time()
l_time=time.time()
cnt=0

while(cap.isOpened()):
    ret, img = cap.read()
    cnt=cnt+1
    if not ret:
        break
    count+=1
    img = np.flip(img, axis=1)
    #Converting from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask1 = mask1 + mask2
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)
    mask2 = cv2.bitwise_not(mask1)
    res1 = cv2.bitwise_and(background, background, mask = mask1)
    res2 = cv2.bitwise_and(img, img, mask = mask2)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
    cv2.imshow(window_name, final_output)
#    cur_time=time.time()
#    print(cur_time-l_time)
#    l_time=cur_time
    k = cv2.waitKey(10)
    if k == 27:
        break
end=time.time()
print(cnt)
print(count)
print(end-start)
cap.release()
cv2.destroyAllWindows()
