import cv2
import numpy as np

# Load the image
image = cv2.imread('images/horse_4.png', cv2.IMREAD_UNCHANGED)

# Check if the image has an alpha channel
if image.shape[2] == 4:
    print("The image has an alpha channel.")
    # Separate the color and alpha channels
    bgr = image[:, :, :3]
    alpha = image[:, :, 3]
    
    # Create a mask using the alpha channel
    mask = alpha > 0

    # Create a new image with the transparent background removed
    result = np.zeros_like(bgr)
    result[mask] = bgr[mask]
    
    # Save or display the result
    cv2.imwrite('horse_image_no_background.png', result)
    cv2.imshow('Image without Background', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("The image does not have an alpha channel.")