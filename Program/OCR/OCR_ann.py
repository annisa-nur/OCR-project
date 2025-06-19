#!/usr/bin/env python
# coding: utf-8
import os
import cv2
import numpy as np
import string
import tensorflow as tf
from path_config import *
from PREPROCESSING.preprocessing_cropped_ann import *
from PREPROCESSING.preprocessing_black_cell import *
import itertools

# Get the same logger instance as the main program
logger = logging.getLogger('__main__')

# ## OCR Running
def ocr_id_ann(final_image, roi_id_checksheet, loaded_roi_id_checksheet):
    # Load model ML (printed)
    try:
        model = tf.keras.models.load_model(ANN_DIGIT_PRINT_PINK)
        logger.info("Model ANN successfully loaded.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e  # Re-raise the error to stop further execution
    
    print("ANN model has been loaded")
    output_folder = ANN_OUTPUT_FOLDER

    proba_final = []

    if roi_id_checksheet.is_file():
        for roi in loaded_roi_id_checksheet:
            x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
            cropped = final_image[y1:y2, x1:x2]
            crop_raw = "crop_raw_id.jpg"
            output_cropped = os.path.join(output_folder, crop_raw)
            cv2.imwrite(output_cropped, cropped)

            # Check if the cropped region is empty
            is_filled = np.any(cropped == 0)  # Cek apakah ada isi di ROI
            if is_filled:
                cropped_processed = preprocessing_cropped_image_ann(output_cropped)  # Preprocessing

                # Prediksi digit
                try:
                    prediction = model.predict(cropped_processed)  # Output: Probabilitas untuk 10 kelas (0-9)
                    prediction = list(itertools.chain(*prediction))
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

    return proba_final

def ocr_pink_digit_print_ann(final_image, roi_digit_print_pink, loaded_roi_digit_print_pink, target_cell_cl, id):
    # Load DNN Model
    try:
        model = tf.keras.models.load_model(ANN_DIGIT_PRINT_PINK)
        logger.info("Printed digit pink model successfully loaded.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e  # Re-raise to stop execution
    output_folder = ANN_OUTPUT_FOLDER

    # proba_list = []
    proba_final = []

    if roi_digit_print_pink.is_file():   
        for i, roi in zip(target_cell_cl, loaded_roi_digit_print_pink):
            print(f"{roi}")
            name, key, x1, y1, x2, y2 = roi["name"], roi["key"], roi["x1"], roi["y1"], roi["x2"], roi["y2"]
            cropped = final_image[y1:y2, x1:x2]
            crop_raw = "crop_raw_print.jpg"
            output_cropped = os.path.join(output_folder, crop_raw)
            cv2.imwrite(output_cropped, cropped)

            output_dir = os.path.join(RESULT_FOLDER, "image/")
            file_name = f"{id}_{name}_{key}_number.jpg"
            output_path = os.path.join(output_dir, file_name)

            height, width = cropped.shape
            total_pixel = width * height
            
            count_black = np.sum(cropped != 255)
            ratio_black = count_black / total_pixel * 100
            
            if ratio_black > 0.5:
                cropped_processed, canvas = preprocessing_cropped_image_ann(output_cropped)  # Preprocessing
                cv2.imwrite(output_path, canvas)
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
                cropped_for_saving, canvas = preprocessing_cropped_image_ann(output_cropped)  # Preprocessing
                cv2.imwrite(output_path, canvas)
                proba_dict = {i: [0] * 10}
                proba_final.append(proba_dict)
                
    return proba_final

def ocr_pink_handw_ann(final_image, roi_handw, loaded_roi_handwritten, target_cell_cl_number, target_cell_cl_text, id):
    # Load DNN Model
    try:
        model_digit = tf.keras.models.load_model(ANN_DIGIT_HANDW_PINK)
        model_text = tf.keras.models.load_model(ANN_LETTER_PINK)
        logger.info("Handwritten model successfully loaded.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e  # Re-raise to stop execution

    proba_final_number = []
    proba_final_letter = []
    roi_number = []
    roi_letter = []
    output_folder = ANN_OUTPUT_FOLDER

    if roi_handw.is_file():    
        for roi in loaded_roi_handwritten:  # Loop dengan indeks (i)
            if roi["type"] == "number":
                roi_number.append(roi)
            else:
                roi_letter.append(roi)

        for roi, target in zip(roi_number, target_cell_cl_number):
            name, key, type, x1, y1, x2, y2 = roi["name"], roi["key"], roi["type"], roi["x1"], roi["y1"], roi["x2"], roi["y2"]
            cropped = final_image[y1:y2, x1:x2]
            crop_raw = "crop_raw_number.jpg"
            output_cropped = os.path.join(output_folder, crop_raw)
            cv2.imwrite(output_cropped, cropped)

            output_dir = os.path.join(RESULT_FOLDER, "image/")
            file_name = f"{id}_{name}_{key}_{type}.jpg"
            output_path = os.path.join(output_dir, file_name)

            # Check if the cropped region is empty
            height, width = cropped.shape
            total_pixel = width * height
            
            count_black = np.sum(cropped != 255)
            ratio_black = count_black / total_pixel * 100
            
            if ratio_black > 1:
                cropped_processed, canvas = preprocessing_cropped_image_ann(output_cropped)  # Preprocessing            
                cv2.imwrite(output_path, canvas)

                # Prediksi digit
                try:
                    prediction = model_digit.predict(cropped_processed)  # Output: Probabilitas untuk 10 kelas (0-9)
                    prediction = list(itertools.chain(*prediction))
                    proba_dict = {target: prediction}
                    proba_final_number.append(proba_dict)
                except Exception as e:
                    print(f"predicting has some issues: {e}")
                    logger.error(f"Predicting has some issues: {e}") 
            else:
                print("Empty cell")
                cropped_for_saving, canvas = preprocessing_cropped_image_ann(output_cropped)  # Preprocessing            
                cv2.imwrite(output_path, canvas)
                proba_dict = {target: [0] * 10}
                proba_final_number.append(proba_dict)
            
        for roi, target in zip(roi_letter, target_cell_cl_text):
            name, key, type, x1, y1, x2, y2 = roi["name"], roi["key"], roi["type"], roi["x1"], roi["y1"], roi["x2"], roi["y2"]
            cropped = final_image[y1:y2, x1:x2]
            crop_raw = "crop_raw_text.jpg"
            output_cropped = os.path.join(output_folder, crop_raw)
            cv2.imwrite(output_cropped, cropped)

            output_dir = os.path.join(RESULT_FOLDER, "image/")
            file_name = f"{id}_{name}_{key}_{type}.jpg"
            output_path = os.path.join(output_dir, file_name)
            
            height, width = cropped.shape
            total_pixel = width * height
            
            count_black = np.sum(cropped != 255)
            ratio_black = count_black / total_pixel * 100
            
            if ratio_black > 1:
                cropped_processed, canvas = preprocessing_cropped_letter_ann(output_cropped)  # Preprocessing         
                cv2.imwrite(output_path, canvas)
                # Prediksi huruf
                try:
                    prediction = model_text.predict(cropped_processed)  # Output: Probabilitas untuk 26 kelas (A-Z)
                    prediction = list(itertools.chain(*prediction))
                    proba_dict = {target: prediction}
                    proba_final_letter.append(proba_dict)
                except Exception as e:
                    print(f"predicting has some issues: {e}")
                    logger.error(f"Predicting has some issues: {e}")
            else:
                print("Empty cell")
                cropped_for_saving, canvas = preprocessing_cropped_letter_ann(output_cropped)  # Preprocessing            
                cv2.imwrite(output_path, canvas)
                proba_dict = {target: [0] * 26}
                proba_final_letter.append(proba_dict)

    return proba_final_number, proba_final_letter