import cv2
import numpy as np
import random as rand


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
    global height_ratio, width_ratio, x, y, w, h
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
        x = x - (w / 2)
        y = y - (h / 2)
    return x, y



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
lego_pieces = {}

while vc.isOpened():
    ret, frame = vc.read()
    imghsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    counter += 1
    
    # UI AND DETECTION OF THE MAIN BOARD
    if board_counter == 0:
        if board_first_iteration:
            print('Place the Main Green Board in the center of the camera feed and press Enter to continue')
            board_first_iteration = False
        else:
            if cv2.waitKey(1) == 13:
                board_counter += 1
            get_main_board(imghsv, frame, kernel, lower_green, upper_green)
            cv2.imshow("Frame", frame)
        continue

    # UI AND DETECTION OF THE PIECES
    if ui_counter == 0:
        if ui_first_iteration:
            print('Remove the Main Board out of the feed and place all of the GREEN, RED, BLUE, and YELLOW pieces in the frame.\nPress Enter to continue.')
            ui_first_iteration = False
        green_detection(imghsv, frame, kernel, lower_green, upper_green, width_ratio, height_ratio)
        red_detection(imghsv, frame, kernel, lower_red1, upper_red1, lower_red2, upper_red2, width_ratio, height_ratio)
        blue_detection(imghsv, frame, kernel, lower_blue, upper_blue, width_ratio, height_ratio)
        yellow_detection(imghsv, frame, kernel, lower_yellow, upper_yellow, width_ratio, height_ratio)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 13:
            ui_counter += 1
        continue

    # DETECTION OF THE PIECES
    green_detection(imghsv, frame, kernel, lower_green, upper_green, width_ratio, height_ratio)
    red_detection(imghsv, frame, kernel, lower_red1, upper_red1, lower_red2, upper_red2, width_ratio, height_ratio)
    blue_detection(imghsv, frame, kernel, lower_blue, upper_blue, width_ratio, height_ratio)
    yellow_detection(imghsv, frame, kernel, lower_yellow, upper_yellow, width_ratio, height_ratio)
    
    # PRINTING THE SIZE OF THE PIECES
    if counter >= 100:
        print(f'RED: {int(red_height_studs)}x{int(red_width_studs)}')
        # print(f'GREEN: {int(green_height_studs)}x{int(green_width_studs)}')
        print(f'BLUE: {int(blue_height_studs)}x{int(blue_width_studs)}')
        print(f'YELLOW: {int(yellow_height_studs)}x{int(yellow_width_studs)}')
        counter = 0

    # UI FOR THE USER TO START THE NESTED WHILE LOOP TO PLACE THE PIECES ON THE BOARD
    if ui_counter == 1:
        print('Press Enter for the program to place the pieces on the board.')
        ui_counter += 1
    if cv2.waitKey(1) == 13:
        dict_first_iteration = True

    # UI FOR THE USER TO RETAKE THE VALUES OF THE PIECES AND THE SET UP OF THE DICIONARY
    if dict_first_iteration:
        print('The pieces have been detected. If there are any wrong values press "r" to retake them.') 
        # MAYBE ADD A FOR LOOP TO ADD THE CORRECT PIECES TO THE DICTIONARY
        lego_pieces = {'red': [int(red_height_studs), int(red_width_studs)], 'blue': [int(blue_height_studs), int(blue_width_studs)], 'yellow': [int(yellow_height_studs), int(yellow_width_studs)]}
        main_board_height = 8
        main_board_width = 16
        main_board = np.zeros((8, 16), dtype=int)
        dict_first_iteration = False

    # SETTING UP THE VARIABLE FOR THE NESTED WHILE LOOP
    next_piece = True
    all_pieces_placed = False
    nested_loop = False
    wait_main_board = True
    nested_counter = 0

    # NESTED WHILE LOOP TO PLACE THE PIECES ON THE BOARD
    while len(lego_pieces) > 0:
        ret, frame = vc.read()
        imghsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        nested_loop = True

        if nested_counter == 0 :
            if wait_main_board:
                print('Place the Main Board back in the frame and press Space to continue.')
                wait_main_board = False
            get_main_board(imghsv, frame, kernel, lower_green, upper_green)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == 32:
                nested_counter += 1
            continue

        board_x, board_y = get_main_board(imghsv, frame, kernel, lower_green, upper_green)

        if next_piece:
            # CHOOSING A RANDOM PIECE FROM THE DICTIONARY
            random_piece = rand.choice(list(lego_pieces.keys()))

            # GETTING A RANDOM LOCATION ON THE BOARD TO PLACE THE PIECE
            piece_height, piece_width = lego_pieces[random_piece]
            random_row = np.random.randint(0, main_board_height - piece_height + 1)
            random_column = np.random.randint(0, main_board_width - piece_width + 1)

            # CHECKING IF THE PIECE FITS ON THE BOARD
            if random_row + piece_height > main_board_height:
                random_row = main_board_height - piece_height
            if random_column + piece_width > main_board_width:
                random_column = main_board_width - piece_width

            # CHECKING IF THE PIECE CAN BE PLACED ON THE BOARD
            if np.sum(main_board[random_row:random_row + piece_height, random_column:random_column + piece_width]) == 0: 
                # REPLACING THE VALUES OF THE PIECE ON THE BOARD WITH 1 TO INDICATE THAT THERE IS A PIECE THERE
                main_board[random_row:random_row + piece_height, random_column:random_column + piece_width] = 1
                cv2.rectangle(frame, (int(board_x + random_column*width_ratio), int(board_y + random_row*height_ratio)), (int(board_x + random_column*width_ratio + piece_width*width_ratio), int(board_y + random_row*height_ratio + piece_height*height_ratio)), (0, 255, 0), 2)
                cv2.putText(frame, random_piece, (int(board_x + random_column*width_ratio), int(board_y + random_row*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 
                print(f'Place the {random_piece} {piece_height}x{piece_width} piece at {random_row + 1}, {random_column + 1} and press Enter to continue')
            else:
                cv2.imshow("Frame", frame)
                continue # NEED TO HAVE THIS HERE SO THAT IT KEEPS GOING THROUGH THE WHILE LOOP UNTIL IT FINDS A PLACE FOR THE PIECE

            # MAKING SURE THAT THE NESTED WHILE LOOP DOESN'T RUN AGAIN UNTIL THE USER CONFIRMS THAT THE PIECE IS PLACED CORRECTLY
            next_piece = False

        board_x, board_y = get_main_board(imghsv, frame, kernel, lower_green, upper_green)

        # KEEP DRAWING THE RECTANGLE AND THE TEXT SO IT'S DRAWN EVERY FRAME -- MAYBE REMOVE THE WIDTH AND HEIGHT RATIO
        cv2.rectangle(frame, (int(board_x + random_column*width_ratio), int(board_y + random_row*height_ratio)), (int(board_x + random_column*width_ratio + piece_width*width_ratio), int(board_y + random_row*height_ratio + piece_height*height_ratio)), (0, 255, 0), 2)
        cv2.putText(frame, random_piece, (int(board_x + random_column*width_ratio), int(board_y + random_row*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 

        cv2.imshow("Frame", frame)
        # MAKE IT GO TO THE NEXT PIECE
        if cv2.waitKey(1) == 13:
            lego_pieces.pop(random_piece)
            next_piece = True
            continue

    if nested_loop:
        all_pieces_placed = True
    if all_pieces_placed:
        print('All the pieces have been placed on the board.')

    # RESET THE VALUES OF THE DICTIONARY
    if cv2.waitKey(1) == 114: 
        dict_first_iteration = True

    cv2.imshow("Frame", frame)
    # FINISH THE PROGRAM
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
vc.release()
