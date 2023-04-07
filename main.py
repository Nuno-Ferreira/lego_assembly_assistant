import cv2
import numpy as np
#import os

# IMAGE FOLDER AND IMAGE NAME
images_folder = r'./lego_images/' # USED FOR ALL OS
lego_image = 'all'

# SETTING UP THE KERNEL AND IMAGE
kernel = np.ones((5, 5), np.uint8)
img = cv2.imread(images_folder + lego_image + '.jpeg')

# GREEN COLOR RANGES
lower_green = np.array([50,50,50])
upper_green = np.array([80,255,255])

# YELLOW COLOR RANGES
lower_yellow = np.array([20,100,100])
upper_yellow = np.array([50,255,255])

# RED COLOR RANGES
lower_red1 = np.array([0,50,50]) # NEED TO ADD THE SECOND MASK TO DETECT RED DUE TO HOW HSV WORKS
upper_red1 = np.array([5,255,255])
lower_red2 = np.array([170,50,50]) 
upper_red2 = np.array([180,255,255])

# BLUE COLOR RANGES
lower_blue = np.array([90,100,100])
upper_blue = np.array([160,255,255])


#------------------------------------------------- YELLOW COLOR DETECTION -----------------------------------------------#

imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_yellow = cv2.inRange(imghsv, lower_yellow, upper_yellow)
img_yellow_mask = cv2.bitwise_and(img, img, mask=mask_yellow)
yellow_mask = cv2.morphologyEx(img_yellow_mask, cv2.MORPH_CLOSE, kernel)
yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

# FOR DETECTING AND DRAWING THE CONTOURS OF THE YELLOW MASK
img_yellow = cv2.cvtColor(yellow_mask, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(img_yellow, 25)
thr_value, img_thresh = cv2.threshold(img_yellow, 100, 200, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
yellow_output = cv2.drawContours(img, contours, -1, (0, 255, 255), 2)

# COLOUR IDENTIFICATION
for i, c in enumerate(contours):
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    area = cv2.contourArea(c)
    cv2.putText(yellow_output, 'YELLOW', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


#------------------------------------------------- GREEN COLOR DETECTION -----------------------------------------------#

mask_green = cv2.inRange(imghsv, lower_green, upper_green)
img_green_mask = cv2.bitwise_and(img, img, mask=mask_green)
green_mask = cv2.morphologyEx(img_green_mask, cv2.MORPH_CLOSE, kernel)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

# FOR DETECTING AND DRAWING THE CONTOURS OF THE GREEN MASK
img_green = cv2.cvtColor(green_mask, cv2.COLOR_BGR2GRAY)
thr_value, img_thresh = cv2.threshold(img_green, 100, 200, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
green_output = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
for i, c in enumerate(contours):
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    area = cv2.contourArea(c)
    cv2.putText(green_output, 'GREEN', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


#------------------------------------------------ RED COLOR DETECTION   -----------------------------------------------#

mask_red = cv2.inRange(imghsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(imghsv, lower_red2, upper_red2)
combined_mask = cv2.bitwise_or(mask_red, mask_red2) # COMBINE THE TWO MASKS TO DETECT RED
img_red_mask = cv2.bitwise_and(img, img, mask=combined_mask)
red_mask = cv2.morphologyEx(img_red_mask, cv2.MORPH_CLOSE, kernel)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

# FOR DETECTING AND DRAWING THE CONTOURS OF THE RED MASK
img_red = cv2.cvtColor(red_mask, cv2.COLOR_BGR2GRAY)
thr_value, img_thresh = cv2.threshold(img_red, 100, 200, cv2.THRESH_BINARY)
contours, hierarchy= cv2.findContours(img_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
red_output = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
for i, c in enumerate(contours):
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    area = cv2.contourArea(c)
    cv2.putText(red_output, 'RED', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


#----------------------------------------------- BLUE COLOR DETECTION   -----------------------------------------------#

mask_blue = cv2.inRange(imghsv, lower_blue, upper_blue)
img_blue_mask = cv2.bitwise_and(img, img, mask=mask_blue)
blue_mask = cv2.morphologyEx(img_blue_mask, cv2.MORPH_CLOSE, kernel)
blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

# FOR DETECTING AND DRAWING THE CONTOURS OF THE BLUE MASK
img_blue = cv2.cvtColor(blue_mask, cv2.COLOR_BGR2GRAY)
thr_value, img_thresh = cv2.threshold(img_blue, 100, 200, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
blue_output = cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
for i, c in enumerate(contours):
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    area = cv2.contourArea(c)
    cv2.putText(blue_output, 'BLUE', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


#---------------------------------------------- IMAGE PROCESSING    --------------------------------------------------#

img = cv2.imread(images_folder + lego_image + '.jpeg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.blur(img_gray, (5, 5))
img_erode = cv2.erode(img_blur, kernel, iterations=2)
img_dilate = cv2.dilate(img_erode, kernel, iterations=2)
thr_value, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # add this: cv2.THRESH_OTSU / OTHER VALUES 50 - 80
img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# DISPLAYING CONTOURS
for i, c in enumerate(contours):
    x,y,w,h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    length = len(contours[i])
    if hierarchy[0,i,3] == -1 and len(contours[i]) >= 200:
        area = cv2.contourArea(c)
        print('area: ', area)
        print('length: ', length)
        # cv2.rectangle(img, (x,y),( x+w,y+h), (0, 255, 0), 2) # USED TO DRAW RECTANGLES AROUND THE PIECES
        cv2.drawContours(img, contours, i, (0, 255, 0), 2) # USED TO DRAW CONTOURS AROUND THE PIECES



# REPLACE THIS BY COMPUTING THE SIZE OF THE MAIN BOARD AND THEN COMPARE IT TO THE SIZE OF THE OTHER PIECES TO GET 4X4/3X3/2X2
# # CIRCLE DETECTION
# circles = cv2.HoughCircles(img_yellow, cv2.HOUGH_GRADIENT, 1, 10, param1=30, param2=50, minRadius=0, maxRadius=1000)

# if circles is not None:
#     detected_circles = np.uint16(np.around(circles))
#     for (x, y, r) in detected_circles[0, :]:
#         cv2.circle(img, (x, y), r, (0, 255, 0), 2)



#------------------------------------- DISPLAYING IMAGES    ----------------------------------------------------------#

cv2.namedWindow('picture', cv2.WINDOW_NORMAL)
# cv2.namedWindow('red mask', cv2.WINDOW_NORMAL)
cv2.imshow('picture', img)
# cv2.imshow('red mask', red_output)
# cv2.imshow('blue mask', blue_output)
# cv2.imshow('green mask', green_output)

key = cv2.waitKey(0)
cv2.destroyAllWindows()
