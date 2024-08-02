# Doing the correct image rescaling and inserting into the segmented area (TOO SLOW TO WORK REAL TIMEE)

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

replacement_image = cv.imread('images/horse_4.png', cv.IMREAD_UNCHANGED)

# Check if the image has an alpha channel
if replacement_image.shape[2] == 4:
    print("The image has an alpha channel.")
else:
    print("The image does not have an alpha channel.")

alpha_channel = replacement_image[:, :, 3]
non_zero_indices = np.where(alpha_channel > 0)
x_min_rep, x_max_rep = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
y_min_rep, y_max_rep = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
cropped_image = replacement_image[y_min_rep:y_max_rep+1, x_min_rep:x_max_rep+1]

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


    # TODO: Might need to replace the object 
    # Draw bounding boxes for moving objects
    for (x1, y1, x2, y2, conf, cls) in moving_detections:
        cv.rectangle(frame_2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{yolo_model.names[int(cls)]} {conf:.2f}"
        cv.putText(frame_2, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)



    #frame_2 = results.render()[0]

    # Assume the first detected object is the one to replace
    x1, y1, x2, y2, conf, cls = map(int, detections[0])
    object_width = x2 - x1
    object_height = y2 - y1

    # GrabCut algorithm for segmentation
    mask = np.zeros(frame_2.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (x1, y1, object_width, object_height)
    cv.grabCut(frame_2, mask, rect, bgd_model, fgd_model, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_area = frame_2 * mask2[:, :, np.newaxis]

    # Extract the bounding box of the segmented area
    y_indices, x_indices = np.where(mask2 == 1)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # Resize the cropped image to minimally cover the segmented area
    segmented_width = x_max - x_min + 1
    segmented_height = y_max - y_min + 1
    resized_image = cv.resize(cropped_image, (segmented_width, segmented_height))

    # Extract the resized alpha channel and RGB channels
    resized_alpha = resized_image[:, :, 3] / 255.0
    resized_rgb = resized_image[:, :, :3]

    # Prepare the region for overlay
    overlay_region = frame_2[y_min:y_max+1, x_min:x_max+1]

    # Apply the mask to the overlay region
    for c in range(3):
        overlay_region[:, :, c] = overlay_region[:, :, c] * (1 - resized_alpha) + resized_rgb[:, :, c] * resized_alpha

    # Place the overlay in the original image
    frame_2[y_min:y_max+1, x_min:x_max+1] = overlay_region

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

    # TODO: Might need to implement options to create 1, 2, 4 widnows output
    # display the resulting frames
    combined_frame = np.vstack((labeled_frame_1, labeled_frame_2))  # Combine two frames horizontally


    cv.imshow('Combined Output', combined_frame)
    # exit
    if cv.waitKey(1) == ord('q'):
        break

# When everything is done, realease the capture
cap.release()
cv.destroyAllWindows()