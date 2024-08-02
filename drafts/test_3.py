# First working prototype with Image being inserted in the bounding box
import numpy as np
import cv2 as cv
import torch

cap = cv.VideoCapture(0)

# Define the label properties
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (255, 255, 255)  # White color
thickness = 2
label_1 = "Original camera output"
label_2 = "Adjusted camera output"
label_height = 40  # Height of the label area
min_movement_area = 500  # Minimum area of movement to consider

# object detection model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# background subtraction method
background_subtractor = cv.createBackgroundSubtractorMOG2()

replacement_image = cv.imread('images/horse_2.png')


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

    frame_2 = frame.copy()
    results = yolo_model(frame_2)
    detections = results.xyxy[0].cpu().numpy()  # xyxy format: (x1, y1, x2, y2, conf, cls)

    fg_mask = background_subtractor.apply(frame_2)

    moving_detections = []
    for *bbox, conf, cls in detections:
        x1, y1, x2, y2 = map(int, bbox)
        roi = fg_mask[y1:y2, x1:x2]
        movement_area = cv.countNonZero(roi)
        if movement_area > min_movement_area:
            moving_detections.append((x1, y1, x2, y2, conf, cls))

    # Draw bounding boxes for moving objects
    for (x1, y1, x2, y2, conf, cls) in moving_detections:
        cv.rectangle(frame_2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{yolo_model.names[int(cls)]} {conf:.2f}"
        cv.putText(frame_2, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    #frame_2 = results.render()[0]

    # Assume the first detected object is the one to replace TODO: create an object with name = person that gets replaced
    x1, y1, x2, y2, conf, cls = map(int, detections[0])
    mask = np.zeros(frame_2.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    frame_2 = cv.inpaint(frame_2, mask, inpaintRadius=3, flags=cv.INPAINT_NS) # NS vs TELEA

    replacement_image = cv.resize(replacement_image, (x2 - x1, y2 - y1))
    frame_2[y1:y2, x1:x2] = replacement_image

    # Get the size of the frames
    height_1, width_1 = frame.shape[:2]
    height_2, width_2 = frame_2.shape[:2]

    # Create a blank image for labels
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