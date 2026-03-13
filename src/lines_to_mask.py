import numpy as np
import cv2
from pathlib import Path



def lines_to_mask(path_upper, path_lower, img_shape):
    """
    Convert upper and lower line TIFFs to a binary mask in-between them.
    """
    # Read the lines
    upper_img = cv2.imread(str(path_upper), cv2.IMREAD_GRAYSCALE)
    lower_img = cv2.imread(str(path_lower), cv2.IMREAD_GRAYSCALE)

    if upper_img is None or lower_img is None:
        return None

    # Get the y-coordinates of the lines for each x (column)
    # Assuming there's only one point per column for each line
    y_upper = np.argmax(upper_img, axis=0)
    y_lower = np.argmax(lower_img, axis=0)

    # Note: argmax returns 0 if no 255 is found. In some cases, the line might be at y=0.
    # We should filter or handle cases where no line is found if necessary.

    mask = np.zeros(img_shape, dtype=np.uint8)

    for x in range(img_shape[1]):
        y_start = y_upper[x]
        y_end = y_lower[x]
        
        # Check if we actually found a pixel in both
        if upper_img[y_start, x] == 255 and lower_img[y_end, x] == 255:
            if y_start < y_end:
                mask[y_start:y_end, x] = 255
            else:
                # Handle cases where line might be swapped or meet
                mask[y_end:y_start, x] = 255

    return mask

PATH_TO_IMAGES = Path("data/for_segmentation/images")
PATH_TO_MASKS = Path("data/for_segmentation/masks")
PATH_TO_LINES = Path("data/for_segmentation/lines")    
PATH_UPPER_LINE = PATH_TO_LINES/"up"
PATH_LOWER_LINE = PATH_TO_LINES/"down"




if __name__ == "__main__":
    # Create mask directory if it doesn't exist
    PATH_TO_MASKS.mkdir(parents=True, exist_ok=True)

    # Process each image
    # Get image files, extracting the number
    image_files = sorted(PATH_TO_IMAGES.glob("*.png"))
    
    for img_path in image_files:
        img_name = img_path.stem
        # The lines are saved as .tif with the same base name as the png
        upper_line_path = PATH_UPPER_LINE / f"{img_name}.tif"
        lower_line_path = PATH_LOWER_LINE / f"{img_name}.tif"

        if upper_line_path.exists() and lower_line_path.exists():
            print(f"Processing {img_name}...")
            # Load image to get shape
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            mask = lines_to_mask(upper_line_path, lower_line_path, img.shape[:2])
            
            if mask is not None:
                output_path = PATH_TO_MASKS / f"mask_{img_name}.png"
                cv2.imwrite(str(output_path), mask)
            else:
                print(f"Failed to create mask for {img_name}")
        else:
            print(f"Lines not found for {img_name}")

