import os
import cv2
import logging
from path_config import *

# Get the same logger instance as the main program
logger = logging.getLogger('__main__')

# Load the image
def load_image(image_path):
    image = cv2.imread(image_path)
    print("tes tes tes")
    return image

# Preprocess image
def preprocessing_image(image, image_path):
    try:
        # Get image shape
        h, w = image.shape[:2]
            
        # Check size condition
        if w > h:
            resized_image = cv2.resize(image, (1754, 1240))
            resized_image = cv2.rotate(resized_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            resized_image = cv2.resize(image, (1240, 1754))

        logger.info("Image successfully resized.")

        # Grayscaling Image
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        _, corrected_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        logger.info("Image successfully converted to binary.")

        # ## Deskewing Image
        # Find contours
        contours, _ = cv2.findContours(corrected_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the largest contour, which is likely the text area
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]

        # Adjust the angle for rotation
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90

        # Get rotation matrix to deskew
        (h, w) = corrected_image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate the bounding box of the rotated image
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])

        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        # Adjust rotation matrix to fit the new bounding box
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2

        # Rotate the image with the new bounding box
        corrected_image = cv2.warpAffine(corrected_image, rotation_matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        logger.info("Image successfully deskewed.")

        # ## Aligment Marks Area
        # Detect contours
        contours, _ = cv2.findContours(corrected_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # check all bounding boxes
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)  # Get bounding box

        # Initialize a list to store filtered contour data
        filtered_contours = []

        # Convert grayscale to BGR
        corrected_image_bgr = cv2.cvtColor(corrected_image, cv2.COLOR_GRAY2BGR)

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Combine all conditions with `or`
            if w > 1000 and h > 1200 and (h>w):
                filtered_contours.append({
                    "id": i,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                })
                print(f"Filtered Contour {i}: x={x}, y={y}, w={w}, h={h}")

        # Visualize filtered contours
        for fc in filtered_contours:
            x, y, w, h = fc["x"], fc["y"], fc["w"], fc["h"]
            cv2.rectangle(corrected_image_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)

        selected_contour = None

        for i, contour in enumerate(filtered_contours):
            # choose bounding boxed with index 0 or 1
            if i == 0 or i == 1:
                selected_contour = contour
                break

        if selected_contour:
            x, y, w, h = selected_contour["x"], selected_contour["y"], selected_contour["w"], selected_contour["h"]

            if w > h:
                temp = w
                w = h
                h = temp
            # Crop image
            final_image = corrected_image[y:y+h, x:x+w]

            # resize image
            final_image = cv2.resize(final_image, (2480, 3508))
            _, final_image = cv2.threshold(final_image, 100, 255, cv2.THRESH_BINARY)

            # Save and display the final image
            print("saving")
            output_dir = PRINTED_FOLDER_PATH
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_cropped.jpg")
            
            cv2.imwrite(output_path, final_image)
            logger.info("Image successfully cropped.")
        else:
            print("Bounding box cant be found.")
            logger.error("Bounding box cant be found.")

        # Detect table lines (horizontal and vertical)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))

        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(final_image, cv2.MORPH_OPEN, horizontal_kernel)

        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(final_image, cv2.MORPH_OPEN, vertical_kernel)

        # Combine horizontal and vertical lines to get table structure
        table_lines = cv2.add(horizontal_lines, vertical_lines)

        # Dilate the table lines to make them thicker
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_lines = cv2.dilate(table_lines, dilation_kernel, iterations=1)

        # Invert the dilated table lines to create a mask
        mask = cv2.bitwise_not(dilated_lines)

        # Remove the table from the original image using the mask
        final_image = cv2.bitwise_and(final_image, mask)

        # ## Invertion Image
        final_image = cv2.bitwise_not(final_image)
        logger.info("Table has successfully deleted.")

    except Exception as e:
        print(f"Error preprocessing printed: {e}")
        logger.error(f"Error preprocessing printed: {e}")
        # return image

    return final_image