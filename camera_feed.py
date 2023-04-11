import cv2
import numpy as np

#--------------------------------- VARIABLES ---------------------------#
url = "http://192.168.1.65:8080/video"
vc = cv2.VideoCapture(url)

kernel = np.ones((5, 5), np.uint8)


# GREEN COLOR RANGES
lower_green = np.array([50,50,50])
upper_green = np.array([80,255,255])


# TELL USER TO PLACE THE MAIN GREEN BOARD IN THE CENTER OF THE CAMERA FEED 
# USE A GREEN MASK TO DETECT THE BIGGEST CONTOUR (MAIN BOARD) -- CONSISTENT LIGHTING AND CAMERA ANGLE
# USE ONE OF THE FRAMES TO CALCULATE THE RATIOS BETWEEN PIXELS AND CM
# USE THAT RATIO TO CALCULATE THE SIZE OF THE LEGO PIECES
# COMPUTE THE AREA OF THE BOARD AND USE IT TO DETECT THE PIECES
# MAIN BOARD = 8x16 


# GET THE MAIN BOARD
def get_main_board(img):
    mask_green = cv2.inRange(img, lower_green, upper_green)
    img_green_mask = cv2.bitwise_and(img, img, mask=mask_green)
    green_mask = cv2.morphologyEx(img_green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # FOR DETECTING AND DRAWING THE CONTOURS OF THE GREEN MASK
    img_green = cv2.cvtColor(green_mask, cv2.COLOR_BGR2GRAY)
    thr_value, img_thresh = cv2.threshold(img_green, 100, 200, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    green_output = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    for cnt in contours:
    # STILL NEED TO CALCULATE THE RATIO BY THE CONTOUR
        rect = cv2.minAreaRect(cnt) # maybe use this
        (x, y), (w, h), angle = rect

        height_ratio = h / 16
        width_ratio = w / 8
    # NEED TO RETURN THE RATIO 

    return ratio


while vc.isOpened():
    ret, frame = vc.read()

    # IMAGE PROCESSING
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(img_gray, (5, 5))
    img_erode = cv2.erode(img_blur, kernel, iterations=2)
    img_dilate = cv2.dilate(img_erode, kernel, iterations=2)
    thr_value, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # add this: cv2.THRESH_OTSU / OTHER VALUES 50 - 80
    img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# INSTRUCT THE USER TO PLACE THE MAIN BOARD IN THE CENTER OF THE CAMERA FEED -- maybe use a button input to confirm it's been placed
# PROCESS THE IMAGE TO DETECT THE MAIN BOARD
    # GETTING THE RATIO TO CALCULATE THE OTHER PIECES
    ratio = get_main_board(frame)

    # GETTING THE WIDTH AND HEIGHT OF THE MAIN BOARD
    board_width = w / ratio
    board_height = h / ratio

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
            cv2.drawContours(frame, contours, i, (0, 255, 0), 2) # USED TO DRAW CONTOURS AROUND THE PIECES


    if frame is not None:
        cv2.imshow("Frame", frame)
    q = cv2.waitKey(1)
    if q==ord("q"):
        break
cv2.destroyAllWindows()

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