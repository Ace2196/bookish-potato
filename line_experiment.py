import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
from math import ceil, fabs

cap = cv2.VideoCapture('./beachVolleyball/beachVolleyball5.mov')
_, img = cap.read()

def isSand(r,g,b):
    h,s,v = rgb_to_hsv(r,g,b)
    return fabs(h-42) < 5 and fabs(s-18) < 20
def isLine(r,g,b):
    h,s,v = rgb_to_hsv(r,g,b)
    return 310 < fabs(h-355) < 330
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

results = np.zeros((img.shape[0], img.shape[1]), np.uint8)
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        b,g,r = img[x][y]
        results[x][y] = isLine(r, g, b)
        
plt.imshow(results)

res = cv2.bitwise_and(img,img, mask=results)

lines = cv2.HoughLines(results, 1, np.pi/180, 200)

for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

plt.figure('test')
plt.imshow(img)