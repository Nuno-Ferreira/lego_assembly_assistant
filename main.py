import cv2
import numpy as np
#import os

images_folder = r'/home/nuno/Documents/lego_assembly_assistant/lego_images/'
lego_image = 'board1'


# vc = cv2.VideoCapture(1)
kernel = np.ones((20, 20), np.uint8)

# GREEN COLOR RANGES
lower_green = np.array([0,100,0])
upper_green = np.array([0,255,0])

# RED COLOR RANGES
lower_red = np.array([0,0,100])
upper_red = np.array([0,0,255])

# BLUE COLOR RANGES
lower_blue = np.array([100,0,0])
upper_blue = np.array([255,0,0])



img = cv2.imread(images_folder + lego_image + '.jpeg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.blur(img_gray, (3, 3))
img_erode = cv2.erode(img_blur, kernel, iterations=1)
img_dilate = cv2.dilate(img_erode, kernel, iterations=1)
thr_value, img_thresh = cv2.threshold(img_blur, 100, 200, cv2.THRESH_BINARY)  # add this: cv2.THRESH_OTSU


# GREEN COLOR DETECTION
imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_green = cv2.inRange(imghsv, lower_green, upper_green)
img_mask = cv2.bitwise_and(img, img, mask=mask_green)

# img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for i, c in enumerate(contours):
    cv2.drawContours(img, contours, i, (0, 255, 0), 2)

# while vc.isOpened():
#     ret, frame = vc.read()
#     #img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     imghsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     #thr_value, img_thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     # B, G, R
#     lower_green = np.array([0,100,0])
#     upper_green = np.array([0,255,0])
#     mask_blue = cv2.inRange(imghsv, lower_green, upper_green)
#     #img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
#     #img_canny = cv2.Canny(img_close, 100, 200)
#     contours, hierarchy = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(frame, contours,-1,(0,255,0),1)


#     cv2.imshow('stream', frame)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break


#cv2.namedWindow('picture', cv2.WINDOW_NORMAL)
cv2.imshow('picture',img)
#cv2.imshow('gray', img_gray)
#cv2.imshow('thresh', img_thresh)
# #cv2.imshow('blur', img_blur)
# #cv2.imshow('close', img_close)
# cv2.imshow('erode', img_erode)
# cv2.imshow('dilate', img_dilate)
cv2.imshow('green mask', img_mask)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.destroyWindow('stream')
# vc.release()
