from sticher import Stitcher
from tracker import Tracker
import argparse
import imutils
import cv2
import os
import progressbar
import utils
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to the first image")
args = vars(ap.parse_args())

VIDEO_NAME = args["video"].split('/')[-1].split('.')[0]
VIDEO_FRAME_DIR = "volley_vid_frames/tracker/%s/"%VIDEO_NAME
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
_, frame_prev = cap.read()
tracker.init(frame_prev)
i = 0
while i < frame_count:
	# bar.update(i)
	_, frame_i = cap.read()
	i += 1
	print(i)
	# while i%STEP_SIZE != 0:
	# 	bar.update(i)
	# 	_, frame_i = cap.read()
	# 	i += 1
	# if i%SPLIT_SIZE==0:
	# 	frame_prev = np.copy(frame_i)
	result = tracker.mean_shift(frame_prev)
	frame_prev = frame_i
	# cv2.imwrite("%s%d.jpg"%(VIDEO_FRAME_DIR,i), result)

