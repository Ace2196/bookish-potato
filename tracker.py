# import the necessary packages
import numpy as np
import cv2
from math import ceil, fabs
import utils
import imutils

class Tracker:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3()

	def track(self, images, ratio=0.75, reprojThresh=4.0,
		showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		(imageB, imageA) = images
		(kpsA, featuresA) = self.detectAndDescribe(imageA)
		(kpsB, featuresB) = self.detectAndDescribe(imageB)

		cv.imshow(removeStandts(img))


		# # match features between the two images
		# M = self.matchKeypoints(kpsA, kpsB,
		# 	featuresA, featuresB, ratio, reprojThresh)

		# # if the match is None, then there aren't enough matched
		# # keypoints to create a panorama
		# if M is None:
		# 	return None

		# # otherwise, apply a perspective warp to stitch the images
		# # together
		# (matches, H, status) = M
		# result = cv2.warpPerspective(imageA, H,
		# 	(imageA.shape[1], imageA.shape[0]))
		# # result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

		# # check to see if the keypoint matches should be visualized
		# if showMatches:
		# 	vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
		# 		status)

		# 	# return a tuple of the stitched image and the
		# 	# visualization
		# 	return (result, vis)

		# # return the stitched image
		# return result

	def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# check to see if we are using OpenCV 3.X
		if self.isv3:
			# detect and extract features from the image
			descriptor = cv2.xfeatures2d.SIFT_create()
			(kps, features) = descriptor.detectAndCompute(image, None)

		# otherwise, we are using OpenCV 2.4.X
		else:
			# detect keypoints in the image
			detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)

			# extract features from the image
			extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)

		# use only keppoints of sand and line
		# print(kps, features)
		court_kps = []
		court_features = []
		for index, kp in enumerate(kps):
			y,x = kp.pt
			b,g,r = image[int(ceil(x)),int(ceil(y))]

			if utils.isSand(r,g,b) or utils.isLine(r,g,b):
			  court_kps.append(kp)
			  court_features.append(features[index])

		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		court_kps = np.float32([kp.pt for kp in court_kps])
		court_features = np.asarray(court_featuresz)

		# return a tuple of keypoints and features
		return (court_kps, court_features)
