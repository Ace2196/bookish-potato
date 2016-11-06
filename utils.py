import cv2


def equalize_color(img):
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])

    img_equalized = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)

    return img_equalized
