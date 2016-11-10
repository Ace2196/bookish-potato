import cv2
from math import ceil, fabs

def isSand(r,g,b):
    h,s,v = rgb_to_hsv(r,g,b)
    return fabs(h-42) < 5 and fabs(s-18) < 20

def isLine(r,g,b):
    h,s,v = rgb_to_hsv(r,g,b)
    return fabs(h-355) < 5

def isInField(r,g,b):
    h,s,v = rgb_to_hsv(r,g,b)
    return fabs(h-42) < 10 and fabs(s-18) < 50

def rgb_to_hsv(r, g, b):
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


def removeStands(img):
    img_blur = cv2.GaussianBlur(img,(31,31),0)
    for x in range(img.shape[0]):
    	for y in range(img.shape[1]):
    		b,g,r = img_blur[x][y]
    		if not isSand(r,g,b):
    			img[x][y] = [255,255,255]
    return img

def blur_image(img):
    img_blur = cv2.GaussianBlur(img,(31,31),0)
    return img_blur

def equalize_color(img):
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])

    img_equalized = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)

    return img_equalized
