import cv2 as cv
import numpy as np

def load_replacement_image(path):
    image = cv.imread(path, cv.IMREAD_UNCHANGED)
    if image.shape[2] != 4:
        raise ValueError("The replacement image does not have an alpha channel.")
    
    alpha_channel = image[:, :, 3]
    non_zero_indices = np.where(alpha_channel > 0)
    x_min, x_max = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
    y_min, y_max = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
    cropped_image = image[y_min:y_max+1, x_min:x_max+1]
    
    return cropped_image

def draw_label(frame, label, font, scale, color, thickness, y_offset=10):
    labeled_frame = np.zeros((frame.shape[0] + 40, frame.shape[1], 3), dtype=np.uint8)
    cv.putText(labeled_frame, label, (10, y_offset), font, scale, color, thickness, cv.LINE_AA)
    labeled_frame[40:, :] = frame
    return labeled_frame