import cv2
import numpy as np
import random as rand
import numpy as np


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


#--------------------------------- FUNCTIONS ---------------------------#

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

        # MAKE SURE THE WIDTH IS ALWAYS GREATER THAN THE HEIGHT
        if h > w:
            h, w = w, h

        # IF THE WIDTH AND HEIGHT ARE GREATER THAN 50, THEN USE THOSE MEASUREMENTS FOR THE RATIOS
        if h and w > 50:
            # CALCULATE THE HEIGHT AND WIDTH RATIOS BASED ON THE HOW MANY STUDS THE MAIN BOARD HAS
            height_ratio = h / 8
            width_ratio = w / 16

            board_output = cv2.drawContours(img, c, -1, (0, 255, 0), 4)



def green_detection(imghsv, img, kernel, lower_green, upper_green, width_ratio, height_ratio):
    global green_width_studs, green_height_studs
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

    for c in contours:

        rect = cv2.minAreaRect(c) 
        (x, y), (w, h), angle = rect

        if h > w:
            h, w = w, h

        red_width_studs = w / width_ratio
        red_height_studs = h / height_ratio

        if h and w > 50:
            red_output = cv2.drawContours(img, c, -1, (0, 0, 255), 4)



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

    for c in contours:

        rect = cv2.minAreaRect(c) 
        (x, y), (w, h), angle = rect

        if h > w:
            h, w = w, h

        blue_width_studs = w / width_ratio
        blue_height_studs = h / height_ratio

        if h and w > 50:
            blue_output = cv2.drawContours(img, c, -1, (255, 0, 0), 4)



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

    for c in contours:
        
        rect = cv2.minAreaRect(c) 
        (x, y), (w, h), angle = rect

        if h > w:
            h, w = w, h

        yellow_width_studs = w / width_ratio
        yellow_height_studs = h / height_ratio

        if h and w > 50:
            yellow_output = cv2.drawContours(img, c, -1, (0, 255, 255), 4)


#------------------MAIN------------------#

print('Starting the while loop...')
counter = 0
board_counter = 0
ui_counter = 0
board_first_iteration = True
ui_first_iteration = True
dict_first_iteration = False

while vc.isOpened():
    # READ THE FRAME AND CONVERT IT TO HSV
    ret, frame = vc.read()
    imghsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # SET THE COUNTER
    counter += 1
    
    if board_counter == 0:
        if board_first_iteration:
            print('Place the main green board in the center of the camera feed and press Enter to continue')
            board_first_iteration = False
        else:
            if cv2.waitKey(1) == 13:
                board_counter += 1
            get_main_board(imghsv, frame, kernel, lower_green, upper_green)
            cv2.imshow("Frame", frame)
        continue

    if ui_counter == 0:
        if ui_first_iteration:
            print('Place all of the GREEN, RED, BLUE, and YELLOW pieces in the frame. \nPress Enter to continue.')
            ui_first_iteration = False
        green_detection(imghsv, frame, kernel, lower_green, upper_green, width_ratio, height_ratio)
        red_detection(imghsv, frame, kernel, lower_red1, upper_red1, lower_red2, upper_red2, width_ratio, height_ratio)
        blue_detection(imghsv, frame, kernel, lower_blue, upper_blue, width_ratio, height_ratio)
        yellow_detection(imghsv, frame, kernel, lower_yellow, upper_yellow, width_ratio, height_ratio)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 13:
            ui_counter += 1
        continue

    get_main_board(imghsv, frame, kernel, lower_green, upper_green)
    green_detection(imghsv, frame, kernel, lower_green, upper_green, width_ratio, height_ratio)
    red_detection(imghsv, frame, kernel, lower_red1, upper_red1, lower_red2, upper_red2, width_ratio, height_ratio)
    blue_detection(imghsv, frame, kernel, lower_blue, upper_blue, width_ratio, height_ratio)
    yellow_detection(imghsv, frame, kernel, lower_yellow, upper_yellow, width_ratio, height_ratio)
    

    if counter >= 100:
        print(f'RED: {int(red_height_studs)}x{int(red_width_studs)}') # NEED TO FIX THIS SO THAT IT PRINTS THE VALUES OF ALL THE PIECES DETECTED AND NOT JUST ONE PIECE OF EACH COLOUR
        print(f'GREEN: {int(green_height_studs)}x{int(green_width_studs)}')
        print(f'BLUE: {int(blue_height_studs)}x{int(blue_width_studs)}')
        print(f'YELLOW: {int(yellow_height_studs)}x{int(yellow_width_studs)}')
        counter = 0

    if ui_counter == 1:
        print('Press Enter for the program to place the pieces on the board.')
        ui_counter += 1
    if cv2.waitKey(1) == 13:
        dict_first_iteration = True

    if dict_first_iteration:
        print('The pieces have been detected. If there are any wrong values press "r" to retake them.') #NEED TO FIX THIS SINCE IT DOESN'T WORK BECAUSE ITS NOT INSIDE THE WHILE LOOP
        lego_pieces = {'red': [red_height_studs, red_width_studs], 'blue': [blue_height_studs, blue_width_studs], 'yellow': [yellow_height_studs, yellow_width_studs]}
        main_board_height = 8
        main_board_width = 16
        main_board = np.zeros((8, 16), dtype=int)
        dict_first_iteration = False

    next_piece = True

    while len(lego_pieces) > 0:
        # MAYBE MAKE A VARIABLE FALSE AT THE BEGINNING OF THE WHILE LOOP AND THEN MAKE IT TRUE WHEN THE PIECE IS PLACED ON THE BOARD (PRESSING ENTER)
        if next_piece:
            print('inside the nested while loop') # THIS IS JUST FOR DEBUG PURPOSES -- DELETE THIS LATER
            random_piece = rand.choice(list(lego_pieces.keys()))

            piece_height, piece_width = lego_pieces[random_piece]
            random_row = np.random.randint(0, main_board_height - piece_height + 1)
            random_column = np.random.randint(0, main_board_width - piece_width + 1)

            if random_row + piece_height > main_board_height:
                random_row = main_board_height - piece_height
            if random_column + piece_width > main_board_width:
                random_column = main_board_width - piece_width

            main_board[random_row:random_row + piece_height, random_column:random_column + piece_width] = 1 # MAYBE PUT THIS INSIDE THE NEXT IF STATEMENT
            
            if np.sum(main_board[random_row:random_row + piece_height, random_column:random_column + piece_width]) == piece_height * piece_width: 
                print('The piece fits on the board')
                cv2.rectangle(frame, (random_column, random_row), (random_column + piece_width, random_row + piece_height), (0, 255, 0), 2) # NEED TO MAKE SURE THAT THE COORDINATES ARE NOT PIXELS BUT RATHER STUDS
                cv2.putText(frame, random_piece, (random_column, random_row), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f'Place the {random_piece} piece at {random_row}, {random_column} and press Enter to continue') # NEED TO FIGURE THIS OUT AND MAKE SURE IT'S ONLY PRINTING ONCE EVERY PIECE IS DRAWN
                lego_pieces.pop(random_piece)
            else:
                print('The piece does not fit on the board')
                main_board[random_row:random_row + piece_height, random_column:random_column + piece_width] = 0
                continue # MAYBE REMOVE THIS

            next_piece = False
            # NEED TO MAKE SURE THAT THE TEXT AND THE DRAWN RECTANGLE IS OUTSIDE THE IF STATEMENT SO THAT IT'S DRAWN EVERY TIME THE WHILE LOOP RUNS
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27:
            break

    # RESET THE VALUES OF THE DICTIONARY
    if cv2.waitKey(1) == 114: 
        dict_first_iteration = True

    cv2.imshow("Frame", frame)
    # FINISH THE PROGRAM
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
vc.release()
