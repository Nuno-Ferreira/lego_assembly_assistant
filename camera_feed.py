import cv2
import numpy as np

url = "http://192.168.1.65:8080/video"
vc = cv2.VideoCapture(url)

kernel = np.ones((5, 5), np.uint8)

while vc.isOpened():
    ret, frame = vc.read()


    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
            cv2.drawContours(frame, contours, i, (0, 255, 0), 2) # USED TO DRAW CONTOURS AROUND THE PIECES


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

    if frame is not None:
        cv2.imshow("Frame", frame)
    q = cv2.waitKey(1)
    if q==ord("q"):
        break
cv2.destroyAllWindows()

# cv2.destroyWindow('stream')
# vc.release()