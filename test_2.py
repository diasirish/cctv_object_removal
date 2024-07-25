import numpy as np
import cv2 as cv
import torch

cap = cv.VideoCapture(0)

# background subtraction method
background_subtractor = cv.createBackgroundSubtractorMOG2()

if not cap.isOpened():
    print("Cannot connect to camera")
    exit()
while True:
    # Capturing Frame by frame
    ret, frame = cap.read()
    # checking if the frame capture is correct
    if not ret:
        print("Can't recieve frame (stream end?). Exiting...")
        break
    fg_mask = background_subtractor.apply(frame)

    cv.imshow('Combined Output', fg_mask)
    # exit
    if cv.waitKey(1) == ord('q'):
        break

# When everything is done, realease the capture
cap.release()
cv.destroyAllWindows()