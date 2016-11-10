from sticher import Stitcher
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

bar = progressbar.ProgressBar(maxval=frame_count, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()
_, frame_prev = cap.read()
i = 0
while i < frame_count:
	bar.update(i)
	_, frame_i = cap.read()
	i += 1
	tracker.track([frame_prev, frame_i])

