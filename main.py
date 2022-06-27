import cv2
import time
import numpy as np
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_file = cv2.VideoWriter("Output.avi",fourcc,20.0,(640,640))

capture = cv2.VideoCapture(0)
time.sleep(2)
bg = 0
for i in range(60):
    ret,bg = capture.read()
bg = np.flip(bg, axis = 1)
while(capture.isOpened()):
    ret,img = capture.read()
    if(ret == False):
        break
    img = np.flip(img, axis = 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_limit = np.array([0,120,50])
    upper_limit = np.array([10,255,255])
    mask_1 = cv2.inRange(hsv,lower_limit,upper_limit)
    lower_limit = np.array([170,120,70])
    upper_limit = np.array([180,255,255])
    mask_2 = cv2.inRange(hsv,lower_limit,upper_limit)
    mask = mask_1 + mask_2
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE, np.ones((3,3), np.uint8))
    new_mask = cv2.bitwise_not(mask)
    result_1 = cv2.bitwise_and(img, img, mask = new_mask)
    result_2 = cv2.bitwise_and(bg,bg,mask = mask)
    final_output = cv2.addWeighted(result_1,1,result_2,1,0)
    output_file.write(final_output)
    cv2.imshow("Cloak",final_output)
    if cv2.waitKey(1)&0xFF == ord("q"):
        break