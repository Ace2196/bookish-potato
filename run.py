from sticher import Stitcher
from tracker import Tracker
import argparse
import imutils
import cv2
import os
# import progressbar
import utils
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to the first image")
args = vars(ap.parse_args())

VIDEO_NAME = args["video"].split('/')[-1].split('.')[0]
VIDEO_FRAME_DIR = "volley_vid_frames/tracker/%s/"%VIDEO_NAME

f = open('%s_p1.txt'%VIDEO_NAME, 'w')

if not os.path.exists(VIDEO_FRAME_DIR):
    os.makedirs(VIDEO_FRAME_DIR)

tracker = Tracker()

cap = cv2.VideoCapture(args["video"])
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)	

i = 0
while i < frame_count:
	_, frame_i = cap.read()

	if i == 0:
		tracker.init(frame_i)

	key, (x, y) = tracker.mean_shift(frame_i)
	print(key)

	# Jump key -> j
	# Use back the previous x, y
	if key == 106:
		(x, y) = (prev_x, prev_y)
		print('jump')
	# Quit key -> q
	# DESTROY ALL WINDOWS
	elif key == 113:
		cv2.destroyAllWindows()
		break
	# + delta y key -> +
	elif key == 61:
		tracker.delta_y += 1
		print(tracker.delta_y)
		continue
	# - delta y key -> -
	elif key == 45:
		tracker.delta_y -= 1
		print(tracker.delta_y)
		continue

	f.write('%d,%d,%d\n'%(i,x,y))
	i += 1
	prev_x, prev_y = (x,y)
	# cv2.imwrite("%s%d.jpg"%(VIDEO_FRAME_DIR,i), result)
f.close()
