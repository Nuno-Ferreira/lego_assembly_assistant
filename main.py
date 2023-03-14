import cv2
import numpy as np
#import os

images_folder = r'/home/nuno/Documents/lego_assembly_assistant/lego_images'

vc = cv2.VideoCapture(1)
kernel = np.ones((5, 5), np.uint8)


while vc.isOpened():
    ret, frame = vc.read()
    #img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imghsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #thr_value, img_thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # B, G, R
    lower_green = np.array([0,100,0])
    upper_green = np.array([0,255,0])
    mask_blue = cv2.inRange(imghsv, lower_green, upper_green)
    #img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
    #img_canny = cv2.Canny(img_close, 100, 200)
    contours, hierarchy = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours,-1,(0,255,0),1)


    cv2.imshow('stream', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyWindow('stream')
vc.release()
