import cv2
import numpy as np
#import os

images_folder = r'/home/nuno/Documents/lego_assembly_assistant/lego_images/'
lego_image = 'all'


kernel = np.ones((5, 5), np.uint8)

# GREEN COLOR RANGES
lower_green = np.array([40,0,0])
upper_green = np.array([80,255,255])

# YELLOW COLOR RANGES
lower_yellow = np.array([20,100,100])
upper_yellow = np.array([50,255,255])

# RED COLOR RANGES
lower_red = np.array([0,0,0])
upper_red = np.array([10,255,255])

# BLUE COLOR RANGES
lower_blue = np.array([90,100,100])
upper_blue = np.array([160,255,255])


# IMAGE PROCESSING
img = cv2.imread(images_folder + lego_image + '.jpeg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.blur(img_gray, (3, 3))
img_erode = cv2.erode(img_blur, kernel, iterations=1)
img_dilate = cv2.dilate(img_erode, kernel, iterations=1)
thr_value, img_thresh = cv2.threshold(img_blur, 100, 200, cv2.THRESH_BINARY)  # add this: cv2.THRESH_OTSU
img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


# GREEN COLOR DETECTION
imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_green = cv2.inRange(imghsv, lower_green, upper_green)
img_green_mask = cv2.bitwise_and(img, img, mask=mask_green)
# green_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)
# green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
# contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# output = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

# RED COLOR DETECTION
mask_red = cv2.inRange(imghsv, lower_red, upper_red)
img_red_mask = cv2.bitwise_and(img, img, mask=mask_red)
# red_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)
# red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
# contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# output = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

# BLUE COLOR DETECTION
mask_blue = cv2.inRange(imghsv, lower_blue, upper_blue)
img_blue_mask = cv2.bitwise_and(img, img, mask=mask_blue)
# blue_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)
# blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
# contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# output = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)


# DISPLAYING CONTOURS
# for i, c in enumerate(contours):
#     cv2.drawContours(img, contours, i, (0, 255, 0), 2)


# DISPLAYING IMAGES
cv2.imshow('picture',img)
#cv2.imshow('gray', img_gray)
#cv2.imshow('thresh', img_thresh)
# #cv2.imshow('blur', img_blur)
# #cv2.imshow('close', img_close)
# cv2.imshow('erode', img_erode)
# cv2.imshow('dilate', img_dilate)
cv2.imshow('green mask', img_green_mask)
cv2.imshow('red mask', img_red_mask)
cv2.imshow('blue mask', img_blue_mask)
# cv2.imshow('red mask', mask)
# cv2.imshow('blue mask', mask)

key = cv2.waitKey(0)
cv2.destroyAllWindows()

