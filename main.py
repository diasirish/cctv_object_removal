import cv2 as cv
import numpy as np
from detectron2.engine import DefaultPredictor
from config import FONT, FONT_SCALE, COLOR, THICKNESS, LABEL_HEIGHT, MIN_MOVEMENT_AREA, get_cfg, update_config_with_args, parse_args
from utils import load_replacement_image, draw_label

def main():
    args = parse_args()
    update_config_with_args(args)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot connect to camera")
        exit()

    cfg = get_cfg()
    predictor = DefaultPredictor(cfg)
    replacement_image = load_replacement_image('images/horse_4.png')
    background_subtractor = cv.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break

        frame_2 = frame.copy()
        fg_mask = background_subtractor.apply(frame_2)

        # Perform inference
        outputs = predictor(frame_2)

        # Get the detected instances
        instances = outputs["instances"]
        boxes = instances.pred_boxes if instances.has("pred_boxes") else None
        scores = instances.scores if instances.has("scores") else None
        classes = instances.pred_classes if instances.has("pred_classes") else None
        masks = instances.pred_masks if instances.has("pred_masks") else None

        # Filter detections based on movement
        moving_detections = []
        for i in range(len(boxes)):
            box = boxes[i].tensor.cpu().numpy().astype(int)[0]
            x1, y1, x2, y2 = box
            conf = scores[i].cpu().numpy()
            cls = classes[i].cpu().numpy()
            mask = masks[i].cpu().numpy().astype(np.uint8)

            roi = fg_mask[y1:y2, x1:x2]
            movement_area = cv.countNonZero(roi)
            if movement_area > MIN_MOVEMENT_AREA:
                moving_detections.append((x1, y1, x2, y2, conf, cls, mask))

        # Draw bounding boxes and replace objects
        for (x1, y1, x2, y2, conf, cls, mask) in moving_detections:
            cv.rectangle(frame_2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {cls} {conf:.2f}"
            cv.putText(frame_2, label, (x1, y1 - 10), FONT, FONT_SCALE, (0, 255, 255), THICKNESS, cv.LINE_AA)

            # Resize the mask to fit the bounding box
            mask_resized = cv.resize(mask, (x2 - x1, y2 - y1), interpolation=cv.INTER_NEAREST)

            # Resize the cropped image to minimally cover the segmented area
            segmented_width = x2 - x1 + 1
            segmented_height = y2 - y1 + 1
            resized_image = cv.resize(replacement_image, (segmented_width, segmented_height))

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
        labeled_frame_1 = np.zeros((height_1 + LABEL_HEIGHT, width_1, 3), dtype=np.uint8)
        labeled_frame_2 = np.zeros((height_2 + LABEL_HEIGHT, width_2, 3), dtype=np.uint8)

        # Add the labels to the blank images
        cv.putText(labeled_frame_1, "Original camera output", (10, LABEL_HEIGHT - 10), FONT, FONT_SCALE, COLOR, THICKNESS, cv.LINE_AA)
        cv.putText(labeled_frame_2, "Adjusted camera output", (10, LABEL_HEIGHT - 10), FONT, FONT_SCALE, COLOR, THICKNESS, cv.LINE_AA)

        # Place the frames below the labels
        labeled_frame_1[LABEL_HEIGHT:LABEL_HEIGHT + height_1, 0:width_1] = frame
        labeled_frame_2[LABEL_HEIGHT:LABEL_HEIGHT + height_2, 0:width_2] = frame_2

        # Display the resulting frames
        combined_frame = np.vstack((labeled_frame_1, labeled_frame_2))  # Combine two frames horizontally
        cv.imshow('Combined Output', combined_frame)

        # Exit
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()