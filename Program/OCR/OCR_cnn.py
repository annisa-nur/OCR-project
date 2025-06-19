#!/usr/bin/env python
# coding: utf-8
import os
import cv2
import numpy as np
import string
from PREPROCESSING.preprocessing_cropped_cnn import *
from path_config import *
import tensorflow as tf
import os
import numpy as np
import cv2
import tensorflow as tf
import itertools

# Get the same logger instance as the main program
logger = logging.getLogger('__main__')

def ocr_id_cnn(final_image, roi_id_checksheet, loaded_roi_id_checksheet):
    # Load model ML (printed)
    try:
        model = tf.keras.models.load_model(CNN_DIGIT_PRINT_PINK)
        logger.info("Model CNN successfully loaded.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e  # Re-raise the error to stop further execution

    print("CNN model has been loaded")
    output_folder = CNN_OUTPUT_FOLDER                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

    proba_final = []
    tc_result = []

    if roi_id_checksheet.is_file():
        for roi in loaded_roi_id_checksheet:
            key, x1, y1, x2, y2 = roi["key"], roi["x1"], roi["y1"], roi["x2"], roi["y2"]
            cropped = final_image[y1:y2, x1:x2]
            tc_result.append(key)

            output_path = os.path.join(output_folder, "crop_1.jpg")
            cv2.imwrite(output_path, cropped)

            # Check if the cropped region is empty
            is_filled = np.any(cropped == 0)  # Cek apakah ada isi di ROI
            if is_filled:
                cropped_processed = preprocessing_cropped_image_cnn(output_path)  # Preprocessing

                # Prediksi digit
                try:
                    prediction = model.predict(cropped_processed)
                    prediction = list(itertools.chain(*prediction))
                    print(prediction)
                    proba_dict = {"CT1": prediction}
                    proba_final.append(proba_dict)
                except Exception as e:
                    print(f"predicting has some issues: {e}")
                    logger.error(f"Predicting has some issues: {e}")
            else:
                print("Empty cell")
                logger.warning(f"Empty ROI for printed at key CT1. Assigning blank and zero on confidence")
                proba_dict = {"CT1": [0] * 10}
                proba_final.append(proba_dict)

    return proba_final, tc_result

def ocr_pink_digit_print_cnn(final_image, roi_digit_print_pink, loaded_roi_digit_print_pink, target_cell_cl):
    # Load CNN Model
    try:
        model = tf.keras.models.load_model(CNN_DIGIT_PRINT_PINK)
        logger.info("Printed digit pink model successfully loaded.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e  # Re-raise to stop execution
    print("CNN model has been loaded")
    output_folder = CNN_OUTPUT_FOLDER

    print(target_cell_cl)
    proba_final = []
    tc_result = []

    if roi_digit_print_pink.is_file():    
        for i, roi in zip(target_cell_cl, loaded_roi_digit_print_pink):
            key, x1, y1, x2, y2 = roi["key"], roi["x1"], roi["y1"], roi["x2"], roi["y2"]
            cropped = final_image[y1:y2, x1:x2]
            tc_result.append(key)

            output_path = os.path.join(output_folder, f"crop_2.jpg")
            cv2.imwrite(output_path, cropped)
            
            height, width = cropped.shape
            total_pixel = width * height
            
            count_black = np.sum(cropped != 255)
            ratio_black = count_black / total_pixel * 100
            
            if ratio_black > 0.5:
                cropped_processed = preprocessing_cropped_image_cnn(output_path)  # Preprocessing

                # Prediksi digit
                try:
                    prediction = model.predict(cropped_processed)  # Output: Probabilitas untuk 10 kelas (0-9)
                    prediction = list(itertools.chain(*prediction))
                    proba_dict = {i: prediction}
                    proba_final.append(proba_dict)
                except Exception as e:
                    print(f"predicting has some issues: {e}")
                    logger.error(f"Predicting has some issues: {e}")
            else:
                print("Empty cell")
                logger.warning(f"Empty ROI for printed at key. Assigning blank and zero on confidence")
                proba_dict = {i: [0] * 10}
                proba_final.append(proba_dict)
                
    return proba_final, tc_result

def ocr_pink_handw_cnn(final_image, roi_handw, loaded_roi_handwritten, target_cell_cl_number, target_cell_cl_text, category=string.ascii_uppercase):
    # Load CNN Model
    try:
        model_digit = tf.keras.models.load_model(CNN_DIGIT_HANDW_PINK)
        model_text = tf.keras.models.load_model(CNN_LETTER_PINK)
        logger.info("Handwritten model successfully loaded.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e  # Re-raise to stop execution

    print(target_cell_cl_number)

    output_folder = CNN_OUTPUT_FOLDER

    proba_final_number = []
    proba_final_letter = []
    tc_result_number = []
    tc_result_letter = []
    roi_number = []
    roi_letter = []

    if roi_handw.is_file():    
        for i, roi in enumerate(loaded_roi_handwritten):  # Loop dengan indeks (i)
            if roi["type"] == "number":
                roi_number.append(roi)
            else:
                roi_letter.append(roi)

        for i, roi in zip(target_cell_cl_number, roi_number):
            key, x1, y1, x2, y2 = roi["key"], roi["x1"], roi["y1"], roi["x2"], roi["y2"]
            cropped = final_image[y1:y2, x1:x2]
            tc_result_number.append(key)
            print(key)

            output_path = os.path.join(output_folder, f"crop_handw_digit_{roi}.jpg")
            cv2.imwrite(output_path, cropped)

            height, width = cropped.shape
            total_pixel = width * height
            
            count_black = np.sum(cropped != 255)
            ratio_black = count_black / total_pixel * 100
            
            if ratio_black > 1:
                cropped_processed = preprocessing_cropped_image_cnn(output_path)  # Preprocessing

                # Prediksi digit
                try:
                    prediction = model_digit.predict(cropped_processed)
                    prediction = list(itertools.chain(*prediction))
                    proba_dict = {i: prediction}
                    proba_final_number.append(proba_dict)
                except Exception as e:
                    print(f"predicting has some issues: {e}")
                    logger.error(f"Predicting has some issues: {e}")  # Output: Probabilitas untuk 10 kelas (0-9)
            else:
                print("Empty cell")
                logger.warning(f"Empty ROI for printed at key. Assigning blank and zero on confidence")
                proba_dict = {i: [0] * 10}
                proba_final_number.append(proba_dict)

        for roi, target in zip(roi_letter, target_cell_cl_text):
            key, x1, y1, x2, y2 = roi["key"], roi["x1"], roi["y1"], roi["x2"], roi["y2"]
            cropped = final_image[y1:y2, x1:x2]
            tc_result_letter.append(key)

            output_path = os.path.join(output_folder, f"crop_handw_letter.jpg")
            cv2.imwrite(output_path, cropped)

            height, width = cropped.shape
            total_pixel = width * height
            
            count_black = np.sum(cropped != 255)
            ratio_black = count_black / total_pixel * 100
            
            if ratio_black > 1:
                cropped_processed = preprocessing_cropped_letter_cnn(output_path)  # Preprocessing
                # logger.info("Roi has been cropped and preprocessed")

                try:
                    prediction = model_text.predict(cropped_processed)  # Output: Probabilitas untuk 10 kelas (0-9)
                    prediction = list(itertools.chain(*prediction))
                    proba_dict = {target: prediction}
                    proba_final_letter.append(proba_dict)
                except Exception as e:
                    print(f"predicting has some issues: {e}")
                    logger.error(f"Predicting has some issues: {e}") 

            else:
                print("Empty cell")
                logger.warning("Empty ROI for printed at key. Assigning blank and zero on confidence")
                proba_dict = {target: [0] * 26}
                proba_final_letter.append(proba_dict)

    return proba_final_number, tc_result_number, proba_final_letter, tc_result_letter
