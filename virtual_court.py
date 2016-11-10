import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to the first image")
args = vars(ap.parse_args())

VIDEO_NAME = args["video"].split('/')[-1].split('.')[0]
VIDEO_FRAME_DIR = "volley_vid_frames/%s/"%VIDEO_NAME

class VirtualCourt(object):
	def __init__(self, court_background_image):
	    # The court specifications are obtained from measurments of the size of a
	    # standard competitive volleyball court
	    # For more info : http://www.livestrong.com/article/362595-dimensions-of-a-beach-volleyball-court/
	    self.real_corners = np.asarray([
	        [50.0,40.0],
	        [470.0,40.0],
	        [470.0,280.0],
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
			real_point = np.dot(vc.homography,np.append(point,1))
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

	def get_homography(self, top_left, top_right, bottom_right, bottom_left):
		image_corners = np.asarray([top_left,top_right,bottom_right,bottom_left])

		M = self.getM(self.real_corners, image_corners)
		U, S, VT = np.linalg.svd(M, full_matrices=True)
		L = VT[-1,:] / VT[-1,-1]
		H = np.asarray(L.reshape(3, 3))

		self.homography = H

if __name__ == '__main__':
	# scheme for points is top_left, top_right, bottom_right, bottom_left
	image_points = np.asarray([[195.0,63.0],[440.0,140.0],[205.0,290.0],[48.0,89.0]])
	test_points = np.asarray([[300,180]])

	vc = VirtualCourt('assets/court_background.png')
	vc.get_homography(*image_points)

	X,Y = vc.getConvertedPoints(test_points)
	vc.show_court(X,Y)
