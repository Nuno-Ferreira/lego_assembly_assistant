import cv2
import numpy as np
import threading
import queue


print("Starting program...")

#--------------------------------- VARIABLES ---------------------------#
url = "http://192.168.1.65:8080/video" # USING THE PHONE AS A WEBCAM
vc = cv2.VideoCapture(1)

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

# DECLARING THE HEIGHT AND WIDTH RATIOS SO THEY CAN BE USED IN THE OTHER FUNCTIONS
width_ratio = 0
height_ratio = 0
green_width_studs = 0
green_height_studs = 0
red_width_studs = 0
red_height_studs = 0
blue_width_studs = 0
blue_height_studs = 0
yellow_width_studs = 0
yellow_height_studs = 0

contours_queue = queue.Queue()
c_queue_lock = threading.Lock()
#--------------------------------- FUNCTIONS ---------------------------#
# TELL USER TO PLACE THE MAIN GREEN BOARD IN THE CENTER OF THE CAMERA FEED AND PRESS 'Q' TO CONTINUE TO THE NEXT STEP
# board_counter = 0
def main_board():
    print('Place the main green board in the center of the camera feed and press Enter to continue')
    input()
    # get_main_board(imghsv, frame, kernel, lower_green, upper_green)
    print('Make sure that the whole main board is selected in the image and then press Enter to continue')
    input()
    # get_main_board(imghsv, frame, kernel, lower_green, upper_green)
    # board_counter += 1


# GET THE MAIN BOARD
def get_main_board(imghsv, img, kernel, lower_green, upper_green):
    
    # DECLARING THE HEIGHT AND WIDTH RATIOS SO THEY CAN BE USED IN THE OTHER FUNCTIONS
    global height_ratio, width_ratio
    while True:
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

            # MAKE SURE THE WIDTH IS ALWAYS GREATER THAN THE HEIGHT
            if h > w:
                h, w = w, h

            # IF THE WIDTH AND HEIGHT ARE GREATER THAN 50, THEN USE THOSE MEASUREMENTS FOR THE RATIOS
            if h and w > 50:
                # CALCULATE THE HEIGHT AND WIDTH RATIOS BASED ON THE HOW MANY STUDS THE MAIN BOARD HAS
                height_ratio = h / 8
                width_ratio = w / 16

                board_output = cv2.drawContours(img, c, -1, (0, 255, 0), 4)
                with c_queue_lock:
                    contours_queue.put(img, c)



def green_detection(imghsv, img, kernel, lower_green, upper_green, width_ratio, height_ratio):
    global green_width_studs, green_height_studs
    while True:
        mask_green = cv2.inRange(imghsv, lower_green, upper_green)
        img_green_mask = cv2.bitwise_and(img, img, mask=mask_green)
        green_mask = cv2.morphologyEx(img_green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        # FOR DETECTING AND DRAWING THE CONTOURS OF THE RED MASK
        img_green = cv2.cvtColor(green_mask, cv2.COLOR_BGR2GRAY)
        # thr_value, img_thresh = cv2.threshold(img_red, 100, 200, cv2.THRESH_BINARY)
        contours, hierarchy= cv2.findContours(img_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for c in contours:

            rect = cv2.minAreaRect(c) 
            (x, y), (w, h), angle = rect

            if h > w:
                h, w = w, h

            green_width_studs = w / width_ratio
            green_height_studs = h / height_ratio

            if h and w > 50:
                green_output = cv2.drawContours(img, c, -1, (0, 0, 255), 4)
                #contours_queue.put(green_output)



def red_detection(imghsv, img, kernel, lower_red1, upper_red1, lower_red2, upper_red2, width_ratio, height_ratio):
    global red_width_studs, red_height_studs
    while True:
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

        for c in contours:

            rect = cv2.minAreaRect(c) 
            (x, y), (w, h), angle = rect

            if h > w:
                h, w = w, h

            red_width_studs = w / width_ratio
            red_height_studs = h / height_ratio

            if h and w > 50:
                red_output = cv2.drawContours(img, c, -1, (0, 0, 255), 4)
                with c_queue_lock:
                    contours_queue.put(img, c)



def blue_detection(imghsv, img, kernel, lower_blue, upper_blue, width_ratio, height_ratio):
    global blue_width_studs, blue_height_studs
    while True:
        mask_blue = cv2.inRange(imghsv, lower_blue, upper_blue)
        img_blue_mask = cv2.bitwise_and(img, img, mask=mask_blue)
        blue_mask = cv2.morphologyEx(img_blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        # FOR DETECTING AND DRAWING THE CONTOURS OF THE BLUE MASK
        img_blue = cv2.cvtColor(blue_mask, cv2.COLOR_BGR2GRAY)
        # thr_value, img_thresh = cv2.threshold(img_blue, 100, 200, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(img_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for c in contours:

            rect = cv2.minAreaRect(c) 
            (x, y), (w, h), angle = rect

            if h > w:
                h, w = w, h

            blue_width_studs = w / width_ratio
            blue_height_studs = h / height_ratio

            if h and w > 50:
                blue_output = cv2.drawContours(img, c, -1, (255, 0, 0), 4)
                with c_queue_lock:
                    contours_queue.put(img, c)



def yellow_detection(imghsv, img, kernel, lower_yellow, upper_yellow, width_ratio, height_ratio):
    global yellow_width_studs, yellow_height_studs
    while True:
        mask_yellow = cv2.inRange(imghsv, lower_yellow, upper_yellow)
        img_yellow_mask = cv2.bitwise_and(img, img, mask=mask_yellow)
        yellow_mask = cv2.morphologyEx(img_yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

        # FOR DETECTING AND DRAWING THE CONTOURS OF THE YELLOW MASK
        img_yellow = cv2.cvtColor(yellow_mask, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(img_yellow, 25)
        thr_value, img_thresh = cv2.threshold(img_yellow, 100, 200, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(img_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for c in contours:
            
            rect = cv2.minAreaRect(c) 
            (x, y), (w, h), angle = rect

            if h > w:
                h, w = w, h

            yellow_width_studs = w / width_ratio
            yellow_height_studs = h / height_ratio

            if h and w > 50:
                yellow_output = cv2.drawContours(img, c, -1, (0, 255, 255), 4)
                with c_queue_lock:
                    contours_queue.put(img, c)



def user_interface():
    print('Place all of the GREEN, RED, BLUE, and YELLOW pieces in the frame. \nPress Enter to continue.') # NEED TO MAKE SURE THIS DOESN'T PRINT OVER AND OVER AGAIN
    input()


def display_info():
    counter = 0
    while vc.isOpened():
        counter += 0.5
        if counter == 1000000:
            print(f'RED: {int(red_height_studs)}x{int(red_width_studs)}')
            print(f'BLUE: {int(blue_height_studs)}x{int(blue_width_studs)}')
            print(f'YELLOW: {int(yellow_height_studs)}x{int(yellow_width_studs)}')
            counter = 0


def display_feed():
    global frame, imghsv
    while vc.isOpened():
        ret, frame = vc.read()
        imghsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow("Frame", frame) # NEED TO SORT OUT THE QUEUE SO THAT THIS CAN LOOP AND SHOW THE FRAME
        # with c_queue_lock:
        #     if not contours_queue.empty():
        #         img, c = contours_queue.get()
        # cv2.drawContours(img, c, -1, (0, 255, 0), 4)
        # cv2.imshow("Contours Queue", img)
        # board_output, red_output, blue_output, yellow_output = contours_queue.get()
        #cv2.imshow("Board", board_output)
        # cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
    vc.release()



def draw_contours():
    with c_queue_lock:
        if not contours_queue.empty():
            img, c = contours_queue.get()
    cv2.drawContours(img, c, -1, (0, 255, 0), 4)
    cv2.imshow("Contours Queue", img)


print('Starting the threads...')

#------------------THREADING------------------#
t_display_feed = threading.Thread(target=display_feed)
t_display_feed.start()
t_main_board = threading.Thread(target=main_board)
t_main_board.start()
t_main_board.join()
t_get_main_board = threading.Thread(target=get_main_board, args=(imghsv, frame, kernel, lower_green, upper_green))
t_get_main_board.start()
t_user_interface = threading.Thread(target=user_interface)
t_user_interface.start()
t_user_interface.join()
t_display_info = threading.Thread(target=display_info)

t_red = threading.Thread(target=red_detection, args=(imghsv, frame, kernel, lower_red1, upper_red1, lower_red2, upper_red2, width_ratio, height_ratio))
t_blue = threading.Thread(target=blue_detection, args=(imghsv, frame, kernel, lower_blue, upper_blue, width_ratio, height_ratio))
t_yellow = threading.Thread(target=yellow_detection, args=(imghsv, frame, kernel, lower_yellow, upper_yellow, width_ratio, height_ratio))


t_red.start()
t_blue.start()
t_yellow.start()
t_display_info.start()

t_red.join()
t_blue.join()
t_yellow.join()
t_display_info.join()
t_get_main_board.join()
t_display_feed.join()



#------------------MAIN------------------#
print('Starting the while loop...')
counter = 0

while vc.isOpened():
    # READ THE FRAME AND CONVERT IT TO HSV
    ret, frame = vc.read()
    imghsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # SET THE COUNTER
    counter += 1
    
    # if board_counter == 0:
    #     main_board(board_counter)

    print('Place all of the GREEN, RED, BLUE, and YELLOW pieces in the frame. \nPress Enter to continue.') # NEED TO MAKE SURE THIS DOESN'T PRINT OVER AND OVER AGAIN
    input()

    red_detection(imghsv, frame, kernel, lower_red1, upper_red1, lower_red2, upper_red2, width_ratio, height_ratio)
    blue_detection(imghsv, frame, kernel, lower_blue, upper_blue, width_ratio, height_ratio)
    yellow_detection(imghsv, frame, kernel, lower_yellow, upper_yellow, width_ratio, height_ratio)

    # SET UP A DICTIONARY TO STORE EACH LEGO PIECE AND ITS DIMENSIONS WITH THE COLOUR BEING THE KEY AND THE DIMENSIONS BEING THE VALUES
    #lego_pieces = {'red': [red_height_studs, red_width_studs], 'blue': [blue_height_studs, blue_width_studs], 'yellow': [yellow_height_studs, yellow_width_studs]} # MAYBE USE THIS OR SOMETHING SIMILAR
    # THEN BY USING RANDOM INTEGERS DRAW ONE OF THE PIECES TO PLACE ON THE MAIN BOARD BY USING IT LIKE COORDINATES AND THEN REMOVE IT FROM THE DICTIONARY

    if counter == 100:
        print(f'RED: {int(red_height_studs)}x{int(red_width_studs)}')
        print(f'BLUE: {int(blue_height_studs)}x{int(blue_width_studs)}')
        print(f'YELLOW: {int(yellow_height_studs)}x{int(yellow_width_studs)}')
        counter = 0

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
vc.release()
