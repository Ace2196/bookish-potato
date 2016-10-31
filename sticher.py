# import the necessary packages
import numpy as np
import cv2
from math import ceil, fabs

class Stitcher:
	def isSand(self,r,g,b):
	    h,s,v = self.rgb_to_hsv(r,g,b)
	    return fabs(h-42) < 5 and fabs(s-18) < 20
	def isLine(self,r,g,b):
	    h,s,v = self.rgb_to_hsv(r,g,b)
	    return fabs(h-355) < 5
	def rgb_to_hsv(self, r, g, b):
	    rr, gg, bb = r/255.0, g/255.0, b/255.0
	    cMax = max(rr, gg, bb)
	    cMin = min(rr, gg, bb)
	    delta = cMax - cMin
	    # Hue calculation
	    if delta == 0:
	        h = 0
	    elif cMax == rr:
	        h = (((gg - bb)/delta)%6)*60
	    elif cMax == gg:
	        h = (((bb - rr)/delta)+2)*60
	    elif cMax == bb:
	        h = (((rr - gg)/delta)+4)*60
	    # Saturation calculation
	    if cMax == 0:
	        s = 0
	    else:
	        s = delta/cMax
	    # Value calculation
	    v = cMax

	    return h, s, v

	def stitch(self, images, ratio=0.75, reprojThresh=4.0,
		showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		(imageB, imageA) = images
		(kpsA, featuresA) = self.detectAndDescribe(imageA)
		(kpsB, featuresB) = self.detectAndDescribe(imageB)

		# match features between the two images
		M = self.matchKeypoints(kpsA, kpsB,
			featuresA, featuresB, ratio, reprojThresh)

		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		if M is None:
			return None

		# otherwise, apply a perspective warp to stitch the images
		# together
		(matches, H, status) = M
		result = cv2.warpPerspective(imageA, H,
			(imageA.shape[1], imageA.shape[0]))
		# result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

		# check to see if the keypoint matches should be visualized
		if showMatches:
			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
				status)

			# return a tuple of the stitched image and the
			# visualization
			return (result, vis)

		# return the stitched image
		return result

	def detectAndDescribe(self, image):
	  # convert the image to grayscale
	  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	  # detect keypoints in the image
	  detector = cv2.FeatureDetector_create("SURF")
	  kps = detector.detect(gray)
	  # mini = min(kps, key=attrgetter('size'))
	  # print(mini.size)
	  court_kps = []
	  for kp in kps:
	      y,x = kp.pt
	      b,g,r = image[int(ceil(x)),int(ceil(y))]

	      if self.isSand(r,g,b) or self.isLine(r,g,b):
	          court_kps.append(kp)

	  # extract features from the image
	  extractor = cv2.DescriptorExtractor_create("SURF")
	  (kps, features) = extractor.compute(gray, court_kps)
	  # convert the keypoints from KeyPoint objects to NumPy
	  # arrays
	  kps = np.float32([kp.pt for kp in kps])

	  # return a tuple of keypoints and features
	  return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])

			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)

			# return the matches along with the homograpy matrix
			# and status of each matched point
			return (matches, H, status)

		# otherwise, no homograpy could be computed
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB

		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

		# return the visualization
		return vis
