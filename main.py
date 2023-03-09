import cv2
import numpy as np
import os

vc = cv2.VideoCapture(0)
kernel = np.ones((5, 5), np.uint8)


while vc.isOpened():
    ret, frame = vc.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thr_value, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
    img_canny = cv2.Canny(img_close, 100, 200)
    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    cv2.imshow('stream', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyWindow('stream')
vc.release()
