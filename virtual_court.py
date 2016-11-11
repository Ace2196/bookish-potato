import matplotlib.pyplot as plt
import numpy as np
import argparse
from json import load as json_load
import cv2
from video import Video
from homography import homography_matrices

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to video")
ap.add_argument("-c", "--corners", required=True,
	help="path to file with hardcoded corner points")
args = vars(ap.parse_args())

VIDEO_NAME = args["video"].split('/')[-1].split('.')[0]
IMAGE_POINTS_FILE = args["corners"]
VIDEO_FRAME_DIR = "volley_vid_frames/%s/"%VIDEO_NAME

class VirtualCourt(object):
	def __init__(self, court_background_image):
	    # The court specifications are obtained from measurments of the size of a
	    # standard competitive volleyball court
	    # For more info : http://www.livestrong.com/article/362595-dimensions-of-a-beach-volleyball-court/
	    self.real_corners = np.asarray([
	        [50.0,40.0],
			[260.0,40.0],
	        [470.0,40.0],
	        [470.0,280.0],
	        [260.0,280.0],
	        [50.0,280.0]
	    ])
	    self.court_size = (520,320)
	    self.court_image = cv2.imread(court_background_image)

	def show_court(self,x,y):
	    plt.scatter(x,y,zorder=1)
	    plt.imshow(self.court_image,zorder=0)
	    plt.axis('off')
	    plt.show()

	def getConvertedPoints(self, points):
		X = []
		Y = []
		for point in points:
			real_point = vc.homography @ np.append(point,1)
			x,y = (real_point[0]/real_point[2], real_point[1]/real_point[2])
			X.append(x)
			Y.append(y)
		return X,Y

	def getM(self, image_points, real_points):
	    M = np.matrix(np.zeros([2*len(image_points),9]))
	    for i in range(len(image_points)):
	        x,y = real_points[i]
	        u,v = image_points[i]
	        M[2*i] = [x, y, 1, 0, 0, 0, -(u*x), -(u*y), -u]
	        M[2*i+1] = [0, 0, 0, x, y, 1, -(v*x), -(v*y), -v]
	    return M

	def get_homography(self, top_left, top_mid, top_right, bottom_right, bottom_mid, bottom_left):
		image_corners = np.asarray([top_left,top_mid,top_right,bottom_right,bottom_mid,bottom_left])

		# M = self.getM(self.real_corners, image_corners)
		# U, S, VT = np.linalg.svd(M, full_matrices=True)
		# L = VT[-1,:] / VT[-1,-1]
		# H = np.asarray(L.reshape(3, 3))

		H, _ = cv2.findHomography(image_corners, vc.real_corners, cv2.RANSAC,
            4.0)

		self.homography = H

if __name__ == '__main__':
	with open(IMAGE_POINTS_FILE) as f:
		hardcoded_points = json_load(f)

	# scheme for points is top_left, top_right, bottom_right, bottom_left
	image_points = hardcoded_points[VIDEO_NAME]
	test_points = [[340,222]]

	video = Video(args["video"])
	# hs = homography_matrices(video)
	hs = np.load('beachVolleyball1.mov.npy')

	vc = VirtualCourt('assets/court_background.png')
	vc.get_homography(*image_points)

	# hs5 = [[1.00183075, .00112675999, -.339743512],
	#  [.000154051517, 1.00083398, -.0622380618],
	#  [.00000128934898, .000000293269759, .999999853]]
	images = video.as_array()
	with open('%s_p%d.txt'%(VIDEO_NAME,1)) as pos:
		for line in pos:
			i,x,y = line.split(',')
			x,y = int(x), int(y)
			p1_point = np.asarray([[int(x)],[int(y)],[1]])
			# print(hs[int(i)])
			pt = hs[int(i)] @ p1_point
			x,y = (pt[0]/pt[2], pt[1]/pt[2])
			img = images[int(i)]
			warped_image = cv2.warpPerspective(
	            img,
	            hs[int(i)],
	            img.shape[1::-1]
	        )
			cv2.rectangle(warped_image,(x-2,y-2),(x+2,y+2),255,thickness=cv2.FILLED)
			cv2.imshow('pic', warped_image)
			k = cv2.waitKey(0)
			if k ==27:
				cv2.destroyAllWindows()
				exit(0)
			X,Y = vc.getConvertedPoints([[int(x),int(y)]])
			print(i)
			vc.show_court(X,Y)
