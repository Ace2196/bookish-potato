from sticher import Stitcher
import argparse
import imutils
import cv2
import os
import progressbar

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
bar = progressbar.ProgressBar(maxval=frame_count, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()
_, frame_prev = cap.read()
i=1
while i < frame_count:
    bar.update(i)
    _, frame_i = cap.read()
    i += 1
    result = stitcher.stitch([frame_prev, frame_i], showMatches=False)
    cv2.imwrite("%s%d.jpg"%(VIDEO_FRAME_DIR,i), result)
