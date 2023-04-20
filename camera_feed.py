import cv2
import numpy as np
import time

#--------------------------------- VARIABLES ---------------------------#
url = "http://192.168.1.65:8080/video" # USING THE PHONE AS A WEBCAM
vc = cv2.VideoCapture(1)

# NEED TO FIX HOW SLOW THE CAMERA FEED IS

# SETTING UP THE KERNEL
kernel = np.ones((10, 10), np.uint8)


# GREEN COLOR RANGES
lower_green = np.array([0,150,100])
upper_green = np.array([90,255,255])

# RED COLOR RANGES
lower_red1 = np.array([0,50,50]) # THE SECOND MASK IS USED TO DETECT THE FULL RANGE OF RED DUE TO HOW HSV WORKS
upper_red1 = np.array([5,255,255])
lower_red2 = np.array([170,50,50]) 
upper_red2 = np.array([180,255,255])

# BLUE COLOR RANGES
lower_blue = np.array([90,100,100])
upper_blue = np.array([160,255,255])

# YELLOW COLOR RANGES
lower_yellow = np.array([20,100,100])
upper_yellow = np.array([50,255,255])

# TELL USER TO PLACE THE MAIN GREEN BOARD IN THE CENTER OF THE CAMERA FEED 

frame_rate = 25
prev = 0

# GET THE MAIN BOARD
def get_main_board(imghsv, img, kernel, lower_green, upper_green):
    
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
    for c in contours:
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
            # print(f'GREEN: {height_ratio, width_ratio}')



def red_detection(imghsv, img, kernel, lower_red1, upper_red1, lower_red2, upper_red2, width_ratio, height_ratio):
    global red_width_studs, red_height_studs

    mask_red = cv2.inRange(imghsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(imghsv, lower_red2, upper_red2)
    combined_mask = cv2.bitwise_or(mask_red, mask_red2) # COMBINE THE TWO MASKS TO DETECT RED
    img_red_mask = cv2.bitwise_and(img, img, mask=combined_mask)
    red_mask = cv2.morphologyEx(img_red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # FOR DETECTING AND DRAWING THE CONTOURS OF THE RED MASK
    img_red = cv2.cvtColor(red_mask, cv2.COLOR_BGR2GRAY)
    # thr_value, img_thresh = cv2.threshold(img_red, 100, 200, cv2.THRESH_BINARY)
    contours, hierarchy= cv2.findContours(img_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # red_output = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    for c in contours:

        rect = cv2.minAreaRect(c) 
        (x, y), (w, h), angle = rect

        if h > w:
            h, w = w, h

        red_width_studs = w / width_ratio
        red_height_studs = h / height_ratio

        if h and w > 50:
            red_output = cv2.drawContours(img, c, -1, (0, 0, 255), 4)
            # print(f'RED: {int(red_height_studs)}x{int(red_width_studs)} \n' f'height: {h} \n' f'width: {w} \n')



def blue_detection(imghsv, img, kernel, lower_blue, upper_blue, width_ratio, height_ratio):
    global blue_width_studs, blue_height_studs

    mask_blue = cv2.inRange(imghsv, lower_blue, upper_blue)
    img_blue_mask = cv2.bitwise_and(img, img, mask=mask_blue)
    blue_mask = cv2.morphologyEx(img_blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    # FOR DETECTING AND DRAWING THE CONTOURS OF THE BLUE MASK
    img_blue = cv2.cvtColor(blue_mask, cv2.COLOR_BGR2GRAY)
    # thr_value, img_thresh = cv2.threshold(img_blue, 100, 200, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # blue_output = cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
    for c in contours:

        rect = cv2.minAreaRect(c) 
        (x, y), (w, h), angle = rect

        if h > w:
            h, w = w, h

        blue_width_studs = w / width_ratio
        blue_height_studs = h / height_ratio

        if h and w > 50:
            blue_output = cv2.drawContours(img, c, -1, (255, 0, 0), 4)
            # print(f'BLUE: {int(blue_height_studs)}x{int(blue_width_studs)} \n' f'height: {h} \n' f'width: {w} \n')


def yellow_detection(imghsv, img, kernel, lower_yellow, upper_yellow, width_ratio, height_ratio):
    global yellow_width_studs, yellow_height_studs

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

    for c in contours:
        
        rect = cv2.minAreaRect(c) 
        (x, y), (w, h), angle = rect

        if h > w:
            h, w = w, h

        yellow_width_studs = w / width_ratio
        yellow_height_studs = h / height_ratio

        if h and w > 50:
            yellow_output = cv2.drawContours(img, c, -1, (0, 255, 255), 4)
            # print(f'YELLOW: {int(yellow_height_studs)}x{int(yellow_width_studs)} \n' f'height: {h} \n' f'width: {w} \n')


# red_prev_state = None
# blue_prev_state = None

counter = 0

while vc.isOpened():
    ret, frame = vc.read()

    imghsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    counter += 1

    get_main_board(imghsv, frame, kernel, lower_green, upper_green)
    red_detection(imghsv, frame, kernel, lower_red1, upper_red1, lower_red2, upper_red2, width_ratio, height_ratio)
    blue_detection(imghsv, frame, kernel, lower_blue, upper_blue, width_ratio, height_ratio)
    yellow_detection(imghsv, frame, kernel, lower_yellow, upper_yellow, width_ratio, height_ratio)

    if counter == 100:
        print(f'RED: {int(red_height_studs)}x{int(red_width_studs)}')
        print(f'BLUE: {int(blue_height_studs)}x{int(blue_width_studs)}')
        print(f'YELLOW: {int(yellow_height_studs)}x{int(yellow_width_studs)}')
        counter = 0

    # if counter == 100:
    #     if red_width_studs and red_height_studs > 0:
    #         print(f'RED: {int(red_height_studs)}x{int(red_width_studs)}')
    #     if blue_width_studs and blue_height_studs > 0:
    #         print(f'BLUE: {int(blue_height_studs)}x{int(blue_width_studs)}')
    #     if yellow_width_studs and yellow_height_studs > 0:
    #         print(f'YELLOW: {int(yellow_height_studs)}x{int(yellow_width_studs)}')
    #     counter = 0

    # red_current_state = red_width_studs, red_height_studs
    # blue_current_state = blue_width_studs, blue_height_studs

    # if red_current_state != red_prev_state:
    #     print(f'RED: {int(red_height_studs)}x{int(red_width_studs)}')
    #     red_prev_state = red_current_state

    # if blue_current_state != blue_prev_state:
    #     print(f'BLUE: {int(blue_height_studs)}x{int(blue_width_studs)}')
    #     blue_prev_state = blue_current_state


    cv2.imshow("Frame", frame)
    q = cv2.waitKey(1)
    if q==ord("q"):
        break
cv2.destroyAllWindows()
vc.release()
# cv2.destroyWindow('stream')
# vc.release()

#--------------------- UNUSED CODE ----------------------------------#

    # #img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # imghsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # #thr_value, img_thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # B, G, R
    # lower_green = np.array([0,100,0])
    # upper_green = np.array([0,255,0])
    # mask_blue = cv2.inRange(imghsv, lower_green, upper_green)
    # #img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
    # #img_canny = cv2.Canny(img_close, 100, 200)
    # contours, hierarchy = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, contours,-1,(0,255,0),1)

    # cv2.imshow('stream', frame)
    # key = cv2.waitKey(1)
    # if key == 27:
    #     break

# def get_main_board(img):
#     mask_green = cv2.inRange(img, lower_green, upper_green)
#     img_green_mask = cv2.bitwise_and(img, img, mask=mask_green)
#     green_mask = cv2.morphologyEx(img_green_mask, cv2.MORPH_CLOSE, kernel)
#     green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

#     # FOR DETECTING AND DRAWING THE CONTOURS OF THE GREEN MASK
#     img_green = cv2.cvtColor(green_mask, cv2.COLOR_BGR2GRAY)
#     thr_value, img_thresh = cv2.threshold(img_green, 100, 200, cv2.THRESH_BINARY)
#     contours, hierarchy = cv2.findContours(img_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     green_output = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

#     for cnt in contours:
#     # STILL NEED TO CALCULATE THE RATIO BY THE CONTOUR
#         rect = cv2.minAreaRect(cnt) # maybe use this
#         (x, y), (w, h), angle = rect

#         height_ratio = h / 16
#         width_ratio = w / 8
#     # NEED TO RETURN THE RATIO 

#     return ratio


    # IMAGE PROCESSING
    # img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    #         cv2.drawContours(frame, contours, i, (0, 255, 0), 2) # USED TO DRAW CONTOURS AROUND THE PIECES

# INSTRUCT THE USER TO PLACE THE MAIN BOARD IN THE CENTER OF THE CAMERA FEED -- maybe use a button input to confirm it's been placed
# PROCESS THE IMAGE TO DETECT THE MAIN BOARD
    # GETTING THE RATIO TO CALCULATE THE OTHER PIECES
    #ratio = get_main_board(frame) #need to change this

    # GETTING THE WIDTH AND HEIGHT OF THE MAIN BOARD
    # board_width = w / ratio
    # board_height = h / ratio