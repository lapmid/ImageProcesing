import numpy as np
import cv2
import time
from imutils import contours
from skimage import measure
import argparse
import imutils


window='Screen'
# Capturing Webcam Feed
cap = cv2.VideoCapture(0)
cv2.namedWindow(window, cv2.WINDOW_NORMAL)

while(True):
	ret, image = cap.read()
	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (11, 11), 0)
	
	thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=4)
	labels = measure.label(thresh, neighbors=8, background=0)
	mask = np.zeros(thresh.shape, dtype="uint8")
	for label in np.unique(labels):
		if label == 0:
			continue
		# otherwise, construct the label mask and count the
		# number of pixels 
		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)

		# if the number of pixels in the component is sufficiently
		# large, then add it to our mask of "large blobs"
		if numPixels > 300:
			mask = cv2.add(mask, labelMask)

	# find the contours in the mask, then sort them from left to
	# right
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	if len(cnts) > 0:
		cnts = contours.sort_contours(cnts)[0]
		print(len(cnts)," source detected")
		# loop over the contours
		for (i, c) in enumerate(cnts):
			# draw the bright spot on the image
			(x, y, w, h) = cv2.boundingRect(c)
			((cX, cY), radius) = cv2.minEnclosingCircle(c)
			cv2.circle(image, (int(cX), int(cY)), int(radius),(0, 0, 255), 3)
			cv2.putText(image, "#{}".format(i + 1), (x, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	# show the output image
	cv2.imshow(window, image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
	# threshold the image to reveal light regions in the
	# blurred image
cap.release()
cv2.destroyAllWindows()