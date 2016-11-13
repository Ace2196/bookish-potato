import matplotlib.pyplot as plt
import numpy as np
import argparse
from json import load as json_load
import cv2
from video import Video
from homography import homography_matrices
import os
import scipy.signal as sigproc
from sys import stdout
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to video")
ap.add_argument("-c", "--corners", required=True,
	help="path to file with hardcoded corner points")
args = vars(ap.parse_args())

VIDEO_NAME = args["video"].split('/')[-1].split('.')[0]
IMAGE_POINTS_FILE = args["corners"]
VIDEO_FRAME_DIR = "volley_vid_frames/tracker/%s/"%VIDEO_NAME

if not os.path.exists(VIDEO_FRAME_DIR):
    os.makedirs(VIDEO_FRAME_DIR)

class VirtualCourt(object):
	def __init__(self, court_background_image):
	    # The court specifications are obtained from measurments of the size of a
	    # standard competitive volleyball court
	    # For more info : http://www.livestrong.com/article/362595-dimensions-of-a-beach-volleyball-court/
	    self.real_corners = np.asarray([
	        [50.0,35.],
			[260.0,35.],
	        [470.0,35.],
	        [470.0,245.],
	        [260.0,245.],
	        [50.0,245.]
	    ])
	    self.scale = (470 - 50)/16
	    self.court_size = (520,320)
	    self.court_image = cv2.imread(court_background_image)

	def true_distance(self, pt1, pt2):
		return np.linalg.norm((pt1 - pt2) * self.scale)

	def show_court(self,x,y):
		plt.scatter([x[0]],[y[0]],color='r')
		plt.scatter([x[1]],[y[1]],color='g')
		plt.scatter([x[2]],[y[2]],color='b')
		plt.scatter([x[3]],[y[3]],color='m')

		plt.imshow(self.court_image,zorder=0)
		plt.axis('off')
		# plt.show()

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
	print('To stitch processed frames to create video, call:')
	print('ffmpeg -framerate %d -i %s%%d.jpg -c:v libx264 -profile:v high -crf 19 -pix_fmt yuv420p %sprocessed_%s.mp4'%(60,VIDEO_FRAME_DIR,VIDEO_FRAME_DIR,VIDEO_NAME))
	print('To stitch plot frames to create video, call:')
	print('ffmpeg -framerate %d -i %splot%%d.png -c:v libx264 -profile:v high -crf 19 -pix_fmt yuv420p %splot_%s.mp4'%(60,VIDEO_FRAME_DIR,VIDEO_FRAME_DIR,VIDEO_NAME))

	video = Video(args["video"])
	images = video.as_array()
	# cv2.imshow('First frame', images[0])
	# k = cv2.waitKey(0)
	# if k ==27:
	# 	cv2.destroyAllWindows()
	# 	exit(0)

	with open(IMAGE_POINTS_FILE) as f:
		hardcoded_points = json_load(f)

	# scheme for points is top_left, top_mid, top_right, bottom_right, bottom_mid, bottom_left
	image_points = hardcoded_points[VIDEO_NAME]

	try:
		hs = np.load('%s_homography.npy'%(VIDEO_NAME))
	except FileNotFoundError:
		hs = homography_matrices(video)
		np.save('%s_homography.npy'%(VIDEO_NAME),hs)

	vc = VirtualCourt('assets/court_background.png')
	vc.get_homography(*image_points)

	player_positions = [[],[],[],[]]
	for pl_num in range(0,4):
		with open('%s_p%d.txt'%(VIDEO_NAME,pl_num+1)) as pos:
			isFirstLine = True
			for line in pos:
				i, x, y = map(
	                lambda c: int(c),
	                line.split(',')
	            )
				if isFirstLine and i>0:
					player_positions[pl_num].append([0,0]*i)
				isFirstLine = False
				point = np.asarray([x,y,1])
				warped_point = np.dot(hs[i], point)
				x = int(warped_point[0]/warped_point[2])
				y = int(warped_point[1]/warped_point[2])
				player_positions[pl_num].append([x,y])
	for pl_num in range(0,4):
		pl_points = player_positions[pl_num]
		pl_x = [point[0] for point in pl_points]
		pl_x = sigproc.savgol_filter(pl_x,15,2)
		pl_y = [point[1] for point in pl_points]
		pl_y = sigproc.savgol_filter(pl_y,15,2)
		player_positions[pl_num] = [[int(pl_x[i]),int(pl_y[i])] for i in range(len(pl_x))]

	point = []
	for i in range(len(player_positions[0])):
		frame_num = i
		stdout.write('{}\r'.format(frame_num))
		stdout.flush()
		pl1_x,pl1_y = player_positions[0][i]
		pl2_x,pl2_y = player_positions[1][i]
		pl3_x,pl3_y = player_positions[2][i]
		pl4_x,pl4_y = player_positions[3][i]

		img = images[frame_num]
		warped_image = cv2.warpPerspective(
            img,
            hs[frame_num],
            img.shape[1::-1]
        )
		cv2.rectangle(warped_image,
						(pl1_x-2,pl1_y-2),(pl1_x+2,pl1_y+2),
						[255,0,0],thickness=cv2.FILLED)
		cv2.rectangle(warped_image,
						(pl2_x-2,pl2_y-2),(pl2_x+2,pl2_y+2),
						[0,0,255],thickness=cv2.FILLED)
		cv2.rectangle(warped_image,
						(pl3_x-2,pl3_y-2),(pl3_x+2,pl3_y+2),
						[255,0,255],thickness=cv2.FILLED)
		cv2.rectangle(warped_image,
						(pl4_x-2,pl4_y-2),(pl4_x+2,pl4_y+2),
						[0,255,255],thickness=cv2.FILLED)
		cv2.imwrite('%s%d.jpg'%(VIDEO_FRAME_DIR,i),warped_image)

		points = [
			[pl1_x,pl1_y],
			[pl2_x,pl2_y],
			[pl3_x,pl3_y],
			[pl4_x,pl4_y]
		]

		X,Y = vc.getConvertedPoints(points)
		vc.show_court(X,Y)
		plt.savefig('%splot%d.png'%(VIDEO_FRAME_DIR,i))
		plt.clf()
