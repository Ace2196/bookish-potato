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
VIDEO_FRAME_DIR = "volley_vid_frames/%s/"%VIDEO_NAME
if not os.path.exists(VIDEO_FRAME_DIR):
    os.makedirs(VIDEO_FRAME_DIR)

stitcher = Stitcher()

cap = cv2.VideoCapture(args["video"])
fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
frame_count = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
print('To stitch frames to create video, call:')
print('ffmpeg -framerate %d -i %s%%d.jpg -c:v libx264 -profile:v high -crf 19 -pix_fmt yuv420p warped%s.mp4'%(fps,VIDEO_FRAME_DIR,VIDEO_NAME))
print('Performing initial sectional mapping')

SPLIT_SIZE = 50
STEP_SIZE = 5

bar = progressbar.ProgressBar(maxval=frame_count, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()
_, frame_prev = cap.read()
i = 0
while i < frame_count:
	bar.update(i)
	_, frame_i = cap.read()
	i += 1
	while i%STEP_SIZE != 0:
		bar.update(i)
		_, frame_i = cap.read()
		i += 1
	if i%SPLIT_SIZE==0:
		frame_prev = np.copy(frame_i)
	if frame_i == None:
		break
	result = stitcher.stitch([frame_prev, frame_i], showMatches=False)
	cv2.imwrite("%s%d.jpg"%(VIDEO_FRAME_DIR,i), result)

print('Performing secondary mapping')
last_section_index = (int(frame_count//SPLIT_SIZE) - 1)*SPLIT_SIZE
bar = progressbar.ProgressBar(maxval=last_section_index, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()
for j in range(last_section_index, -1, -SPLIT_SIZE):
	bar.update(last_section_index-j)
	parentImage = cv2.imread("%s%d.jpg"%(VIDEO_FRAME_DIR,j))
	if parentImage == None:
		break
	images=[]
	for i in range(j+SPLIT_SIZE,int(frame_count),STEP_SIZE):
		img = cv2.imread("%s%d.jpg"%(VIDEO_FRAME_DIR,i))
		if img == None:
			continue
		images.append(img)
	results = stitcher.stichSet(parentImage,images)
	image_tracker = j+SPLIT_SIZE
	if results == None:
		continue
	for res in results:
		cv2.imwrite("%s%d.jpg"%(VIDEO_FRAME_DIR,image_tracker), res)
		image_tracker += STEP_SIZE
