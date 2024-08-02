import numpy as np
import cv2 as cv
import torch
from ultralytics import YOLO

# Initialize the video capture
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

# Load YOLOv8 model
model = YOLO("yolov8n-seg.pt")  # Replace with the appropriate YOLOv8 model path

# Background subtraction method
background_subtractor = cv.createBackgroundSubtractorMOG2()

# Load the replacement image with alpha channel
replacement_image = cv.imread('images/horse_4.png', cv.IMREAD_UNCHANGED)
if replacement_image.shape[2] == 4:
    print("The image has an alpha channel.")
else:
    raise ValueError("The replacement image does not have an alpha channel.")

alpha_channel = replacement_image[:, :, 3]
non_zero_indices = np.where(alpha_channel > 0)
x_min_rep, x_max_rep = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
y_min_rep, y_max_rep = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
cropped_image = replacement_image[y_min_rep:y_max_rep+1, x_min_rep:x_max_rep+1]

if not cap.isOpened():
    print("Cannot connect to camera")
    exit()

while True:
    # Capturing frame by frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    frame_2 = frame.copy()

    # Perform inference with YOLOv8
    results = model(frame_2)

    # Process detections
    detections = results.pred[0]  # First image in the batch
    boxes = detections.boxes
    masks = detections.masks

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        conf = float(box[4])
        cls = int(box[5])
        mask = masks[i].cpu().numpy().astype(np.uint8)

        # Filter detections by confidence and movement area
        roi = mask[y1:y2, x1:x2]
        movement_area = cv.countNonZero(roi)
        if movement_area > min_movement_area:
            cv.rectangle(frame_2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {cls} {conf:.2f}"
            cv.putText(frame_2, label, (x1, y1 - 10), font, font_scale, (0, 255, 255), thickness, cv.LINE_AA)

            # Resize the mask to fit the bounding box
            mask_resized = cv.resize(mask, (x2 - x1, y2 - y1), interpolation=cv.INTER_NEAREST)

            # Resize the cropped image to minimally cover the segmented area
            segmented_width = x2 - x1 + 1
            segmented_height = y2 - y1 + 1
            resized_image = cv.resize(cropped_image, (segmented_width, segmented_height))

            # Extract the resized alpha channel and RGB channels
            resized_alpha = resized_image[:, :, 3] / 255.0
            resized_rgb = resized_image[:, :, :3]

            # Prepare the region for overlay
            overlay_region = frame_2[y1:y2+1, x1:x2+1]

            # Ensure the dimensions match before blending
            if overlay_region.shape[:2] == resized_alpha.shape:
                for c in range(3):
                    overlay_region[:, :, c] = overlay_region[:, :, c] * (1 - resized_alpha) + resized_rgb[:, :, c] * resized_alpha
                # Place the overlay in the original image
                frame_2[y1:y2+1, x1:x2+1] = overlay_region

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

    # Display the resulting frames
    combined_frame = np.vstack((labeled_frame_1, labeled_frame_2))  # Combine two frames horizontally
    cv.imshow('Combined Output', combined_frame)

    # Exit
    if cv.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv.destroyAllWindows()