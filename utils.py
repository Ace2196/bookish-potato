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

def imageColorClustering(img, k):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Z = img.reshape((-1,3))
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = k
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)

def removeStands(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Z = img.reshape((-1,3))
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    res3 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
    res3 = cv2.GaussianBlur(res3,(71,71),0)
    res3 = cv2.cvtColor(res3, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(res3, np.array([133,0,0]), np.array([153,255,255]))
    mask = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(img,img, mask= mask)
    #
    res2 = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    # res2 = cv2.GaussianBlur(res2,(131,131),0)
    return res2

def blur_image(img):
    img_blur = cv2.GaussianBlur(img,(31,31),0)
    return img_blur

def equalize_color(img):
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])

    img_equalized = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)

    return img_equalized
