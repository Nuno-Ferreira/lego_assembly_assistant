import cv2
import numpy as np

# IMAGE FOLDER AND IMAGE NAME
images_folder = r'./lego_images/' # USED FOR ALL OS
lego_image = 'main_board_pieces'

# SETTING UP THE KERNEL AND IMAGE
kernel = np.ones((10, 10), np.uint8)
img = cv2.imread(images_folder + lego_image + '.jpg')
imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# SETTING UP THE HEIGHT AND WIDTH RATIOS TO CALCULATE THE NUMBER OF STUDS
# height_ratio = 1 # maybe can delete these
# width_ratio = 1

# GREEN COLOR RANGES
lower_green = np.array([0,150,100])
upper_green = np.array([90,255,255])

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


#------------------------------------------------- GREEN COLOR DETECTION -----------------------------------------------#

def green_detection(imghsv, img, kernel, lower_green, upper_green):
    
    # DECLARING THE HEIGHT AND WIDTH RATIOS SO THEY CAN BE USED IN THE OTHER FUNCTIONS
    global height_ratio, width_ratio

    mask_green = cv2.inRange(imghsv, lower_green, upper_green)
    img_green_mask = cv2.bitwise_and(img, img, mask=mask_green)
    green_mask = cv2.dilate(img_green_mask, kernel, iterations=2)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.erode(green_mask, kernel, iterations=1)
    # green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # FOR DETECTING AND DRAWING THE CONTOURS OF THE GREEN MASK
    img_green = cv2.cvtColor(green_mask, cv2.COLOR_BGR2GRAY)
    thr_value, img_thresh = cv2.threshold(img_green, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i, c in enumerate(contours):
        # FIND THE MINIMUM AREA RECTANGLE OF THE CONTOUR
        rect = cv2.minAreaRect(c) 
        (x, y), (w, h), angle = rect

        # MAYBE MAKE THIS A SEPARATE FUNCTION SO I CAN USE IT IN OTHER COLOUR FUNCTIONS
        # MAKE SURE THE WIDTH IS ALWAYS GREATER THAN THE HEIGHT
        if h > w:
            h, w = w, h

        # IF THE WIDTH AND HEIGHT ARE GREATER THAN 50, THEN USE THOSE MEASUREMENTS FOR THE RATIOS
        if h and w > 50:
            # CALCULATE THE HEIGHT AND WIDTH RATIOS BASED ON THE HOW MANY STUDS THE MAIN BOARD HAS
            height_ratio = h / 8
            width_ratio = w / 16

            green_output = cv2.drawContours(img, c, -1, (0, 255, 0), 4)
            print(f'GREEN: {height_ratio, width_ratio}')

# NEED TO ADD A RETURN STATEMENT TO RETURN THE HEIGHT AND WIDTH RATIOS TO BE USED IN THE OTHER FUNCTIONS
    # return height_ratio, width_ratio # MAYBE DELETE THIS

"""
If the goal is to identify the lego board regardless of its orientation, 
you may need to modify the code to detect the dimensions of the board in
a more orientation-independent way. One possible approach is to calculate 
the aspect ratio of the bounding rectangle (w/h or h/w) and use that to 
identify the board.
"""

#------------------------------------------------ RED COLOR DETECTION   -----------------------------------------------#

def red_detection(imghsv, img, kernel, lower_red1, upper_red1, lower_red2, upper_red2, width_ratio, height_ratio):
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
    # red_output = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    for i, c in enumerate(contours):

        rect = cv2.minAreaRect(c) 
        (x, y), (w, h), angle = rect

        if h > w:
            h, w = w, h

        width_studs = w / width_ratio
        height_studs = h / height_ratio

        if h and w > 50:
            red_output = cv2.drawContours(img, c, -1, (0, 0, 255), 4)
            print(f'RED: {int(height_studs), int(width_studs)} \n' f'height: {h} \n' f'width: {w} \n')

# #----------------------------------------------- BLUE COLOR DETECTION   -----------------------------------------------#

def blue_detection(imghsv, img, kernel, lower_blue, upper_blue, width_ratio, height_ratio):
    mask_blue = cv2.inRange(imghsv, lower_blue, upper_blue)
    img_blue_mask = cv2.bitwise_and(img, img, mask=mask_blue)
    blue_mask = cv2.morphologyEx(img_blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    # FOR DETECTING AND DRAWING THE CONTOURS OF THE BLUE MASK
    img_blue = cv2.cvtColor(blue_mask, cv2.COLOR_BGR2GRAY)
    thr_value, img_thresh = cv2.threshold(img_blue, 100, 200, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # blue_output = cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
    for i, c in enumerate(contours):

        rect = cv2.minAreaRect(c) 
        (x, y), (w, h), angle = rect

        if h > w:
            h, w = w, h

        width_studs = w / width_ratio
        height_studs = h / height_ratio

        if h and w > 50:
            blue_output = cv2.drawContours(img, c, -1, (255, 0, 0), 4)
            print(f'BLUE: {int(height_studs), int(width_studs)} \n' f'height: {h} \n' f'width: {w} \n')


#------------------------------------------------- YELLOW COLOR DETECTION -----------------------------------------------#

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

#---------------------------------------------- IMAGE PROCESSING    --------------------------------------------------#

# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_blur = cv2.blur(img_gray, (5, 5))
# img_erode = cv2.erode(img_blur, kernel, iterations=2)
# img_dilate = cv2.dilate(img_erode, kernel, iterations=2)
# thr_value, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # add this: cv2.THRESH_OTSU / OTHER VALUES 50 - 80
# img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
# contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# # DISPLAYING CONTOURS
# for i, c in enumerate(contours):
#     x,y,w,h = cv2.boundingRect(c)
#     area = cv2.contourArea(c)
#     length = len(contours[i])
#     if hierarchy[0,i,3] == -1 and len(contours[i]) >= 200:
#         area = cv2.contourArea(c)
#         print('area: ', area)
#         print('length: ', length)
#         # cv2.rectangle(img, (x,y),( x+w,y+h), (0, 255, 0), 2) # USED TO DRAW RECTANGLES AROUND THE PIECES
#         cv2.drawContours(img, contours, i, (0, 255, 0), 2) # USED TO DRAW CONTOURS AROUND THE PIECES


green_detection(imghsv, img, kernel, lower_green, upper_green)
red_detection(imghsv, img, kernel, lower_red1, upper_red1, lower_red2, upper_red2, width_ratio, height_ratio)
blue_detection(imghsv, img, kernel, lower_blue, upper_blue, width_ratio, height_ratio)

#------------------------------------- DISPLAYING IMAGES    ----------------------------------------------------------#

cv2.namedWindow('picture', cv2.WINDOW_NORMAL)
# cv2.namedWindow('red mask', cv2.WINDOW_NORMAL)
cv2.imshow('picture', img)
# cv2.imshow('red mask', red_output)
# cv2.imshow('blue mask', blue_output)
# cv2.imshow('green mask', green_output)

key = cv2.waitKey(0)
cv2.destroyAllWindows()



#------------------------------------- UNUSED CODE ----------------------------------------------------------#

# REPLACE THIS BY COMPUTING THE SIZE OF THE MAIN BOARD AND THEN COMPARE IT TO THE SIZE OF THE OTHER PIECES TO GET 4X4/3X3/2X2
# # CIRCLE DETECTION
# circles = cv2.HoughCircles(img_yellow, cv2.HOUGH_GRADIENT, 1, 10, param1=30, param2=50, minRadius=0, maxRadius=1000)

# if circles is not None:
#     detected_circles = np.uint16(np.around(circles))
#     for (x, y, r) in detected_circles[0, :]:
#         cv2.circle(img, (x, y), r, (0, 255, 0), 2)