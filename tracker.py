# import the necessary packages
import numpy as np
import cv2
from math import ceil, fabs
import utils
import imutils

class Tracker:
	roi = None
	hsv_roi = None
	mask = None
	roi_hist = None
	# Starting position for video 1
	# r, h, c, w = 175, 30, 488, 30 # green-back
	# r, h, c, w = 80, 30, 200, 30 # green-front
	r, h, c, w = 55, 30, 160, 30 # white-right

	# Setup the termination criteria, either 50 iteration or move by atleast 1 pt
	term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1)
	track_window = (c, r, w, h)
	manual_tracking = False

	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3()

	def draw_contours(self, image):
		img = image.copy()
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(gray, 127, 255, 0)
		_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		filtered_contours = []
		for contour in contours:
			# print(cv2.contourArea(contour))
			# Remove contours on sand 
			if cv2.contourArea(contour) > 60 and cv2.contourArea(contour) < 1000:
				filtered_contours.append(contour)
		cv2.drawContours(img, filtered_contours, -1, (0,255,0), 3)
		cv2.drawContours(image, contours, -1, (0,255,0), 3)
		cv2.imshow("img1", img)
		cv2.imshow("imgage", image)
		cv2.waitKey()

		return img


	def init(self, image):
		print '====================== MeanShift: init ==================================='
		# set up the ROI for tracking
		cv2.imshow("img", image)
		img = image.copy()
		# image = utils.blur_image(image)
		_, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
		image = self.draw_contours(image)
		roi = image[self.r: self.r + self.h, self.c: self.c + self.w]
		cv2.imshow("roi",roi)
		cv2.waitKey()

		hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
		# Color of the object to recognize
		mask = cv2.inRange(hsv_roi, np.array((60., 0., 0.)), np.array((180., 255., 255.)))
		roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
		cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

		self.roi = roi
		self.hsv_roi = hsv_roi
		self.mask = mask
		self.roi_hist = roi_hist

		self.draw_image(img)

	def mean_shift(self, image):
		img = image.copy()
		# image = utils.blur_image(image)
		# Remove noise and shadows
		_, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
		image = self.draw_contours(image)

		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

		# apply meanshift to get the new location
		ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)

		image = self.draw_image(img)
		self.draw_dst(dst)
		return image

	def draw_image(self, image):
		x, y, w, h = self.track_window
		image = cv2.rectangle(image, (x,y), (x+w,y+h), 255,2)
		cv2.imshow("image", image)
		cv2.waitKey()
		return image

	def draw_dst(self, image):
		x, y, w, h = self.track_window
		image = cv2.rectangle(image, (x,y), (x+w,y+h), 255,2)
		cv2.imshow("dst", image)
		cv2.waitKey()
		return image

