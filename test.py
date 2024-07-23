import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# Define the label properties
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (255, 255, 255)  # White color
thickness = 2
label_1 = "Original camera input"
label_2 = "Adjusted camera input"

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
    # operations on the frame should be done here
    frame_2 = frame.copy()

    # Get the size of the frames
    height_1, width_1 = frame.shape[:2]
    height_2, width_2 = frame_2.shape[:2]

    # Create a blank image for labels
    label_height = 40  # Height of the label area
    labeled_frame_1 = np.zeros((height_1 + label_height, width_1, 3), dtype=np.uint8)
    labeled_frame_2 = np.zeros((height_2 + label_height, width_2, 3), dtype=np.uint8)

    # Add the labels to the blank images
    cv.putText(labeled_frame_1, label_1, (10, label_height - 10), font, font_scale, color, thickness, cv.LINE_AA)
    cv.putText(labeled_frame_2, label_2, (10, label_height - 10), font, font_scale, color, thickness, cv.LINE_AA)

    # Place the frames below the labels
    labeled_frame_1[label_height:label_height + height_1, 0:width_1] = frame
    labeled_frame_2[label_height:label_height + height_2, 0:width_2] = frame_2

    # display the resulting frames
    combined_frame = np.vstack((labeled_frame_1, labeled_frame_2))  # Combine two frames horizontally
    cv.imshow('Combined Output', combined_frame)
    # exit
    if cv.waitKey(1) == ord('q'):
        break

# When everything is done, realease the capture
cap.release()
cv.destroyAllWindows()