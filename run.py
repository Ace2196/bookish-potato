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

SPLIT_SIZE = 50
STEP_SIZE = 5


# bar = progressbar.ProgressBar(maxval=frame_count, \
#     widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()
i = 0
while i < frame_count:
	_, frame_i = cap.read()

	if i == 0:
		tracker.init(frame_i)

	(x, y) = tracker.mean_shift(frame_i)
	f.write('%d,%d,%d\n'%(i,x,y))
	i += 1
	# cv2.imwrite("%s%d.jpg"%(VIDEO_FRAME_DIR,i), result)
f.close()
