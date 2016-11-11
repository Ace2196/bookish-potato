# import the necessary packages
import numpy as np
import cv2
from math import ceil, fabs
import utils
import imutils

class Tracker:
	image = None
	roi = None
	hsv_roi = None
	mask = None
	roi_hist = None

	r, h, c, w = 150, 20, 450, 30 
	delta_x = 10
	delta_y = 70 # green-back
	# delta_y = 25
	box_x = 10
	box_y = 10

	# Setup the termination criteria, either 50 iteration or move by atleast 1 pt
	term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
	track_window = (c, r, w, h)
	manual_tracking = False

	def mouseEventCallback(self, event, x, y, flags, user_data):
		# print(x, y)
		if event == cv2.EVENT_LBUTTONDOWN:
			# print(x, y)
			self.set_track_window((x-15, y-15, self.w, self.h))
			self.set_roi()
			self.manual_tracking = True

	def set_track_window(self, track_window):
		self.track_window = track_window

	def set_roi(self, image=None):
		# set up the ROI for tracking
		if image == None:
			image = self.image
		c, r, w, h = self.track_window
		self.roi = image[r: r+h, c: c+w]
		cv2.imshow("roi", self.roi)

		return self.roi

	def draw_contours(self, image):
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(gray, 127, 255, 0)
		_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# filter out contours of small size, which are likely to be noise
		filtered_contours = []
		for contour in contours:
			# Remove contours on sand 
			if cv2.contourArea(contour) > 50 and cv2.contourArea(contour) < 1000:
				filtered_contours.append(contour)
		cv2.drawContours(image, filtered_contours, -1, (0,255,0), 3)
		cv2.imshow("contours", image)

		return image

	def setup_mean_shift(self, image):
		roi = self.set_roi(image)
		hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv_roi, np.array((60., 0., 0.)), np.array((180., 255., 255.)))
		roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
		cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

		self.hsv_roi = hsv_roi
		self.mask = mask
		self.roi_hist = roi_hist

	def init(self, image):
		self.image = image.copy()
		# set up player to be tracked
		cv2.imshow("init image", image)
		cv2.setMouseCallback("init image", self.mouseEventCallback)
		cv2.waitKey()

		# preprocess image
		_, image = cv2.threshold(image, 155, 255, cv2.THRESH_BINARY)
		self.draw_contours(image)
		cv2.destroyAllWindows()

		if self.manual_tracking == True:
			self.setup_mean_shift(image)
			self.manual_tracking = False

		self.draw_image(self.image)

	def mean_shift(self, original_image):
		self.image = original_image.copy()
		# preprocess image
		_, image = cv2.threshold(original_image, 155, 255, cv2.THRESH_BINARY)
		self.draw_contours(image)

		# calculate back project
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

		# apply meanshift to get the new location
		ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)

		self.draw_dst(dst)
		(res_image, x, y) = self.draw_image(self.image)

		cv2.setMouseCallback("image", self.mouseEventCallback)
		cv2.waitKey()

		if self.manual_tracking == True:
			self.setup_mean_shift(image)
			self.manual_tracking = False
			return self.mean_shift(original_image)

		return (x, y)

	def draw_image(self, image):
		res_image = image.copy()
		x, y, w, h = self.track_window
		x = x + self.delta_x
		y = y + self.delta_y
		cv2.rectangle(res_image, (x, y), (x+self.box_x, y+self.box_y), 255, thickness=cv2.FILLED)

		# Display updated image to check correctness
		cv2.imshow("image", res_image)
		return (res_image, x+self.box_x/2, y+self.box_y/2)

	def draw_dst(self, image):
		x, y, w, h = self.track_window
		image = cv2.rectangle(image, (x,y), (x+w,y+h), 255,2)
		cv2.imshow("dst", image)
		return image

