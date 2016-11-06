import cv2


def equalize_color(img):
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    channels = cv2.split(img_ycrcb)
    channels[0] = cv2.equalizeHist(channels[0])

    img_equalized = cv2.merge(channels, img)
    img_equalized = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)

    return img_equalized
