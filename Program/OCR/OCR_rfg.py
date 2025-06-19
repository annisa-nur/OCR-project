#!/usr/bin/env python
# coding: utf-8
import os
import cv2
import numpy as np
import joblib
from path_config import *
from pathlib import Path
from OCR_PROGRAM.train_model_handw_gemini import *
from PREPROCESSING.preprocessing_cropped_rfg import *
from PREPROCESSING.preprocessing_printed import *

# ## OCR Running
def ocr_id_rfg(
    final_image,
    roi_id_checksheet, 
    loaded_roi_id_checksheet
):
    # Load model ML (printed)
    model = joblib.load(Path(IDENTIFIER_MODEL))

    text_id_checksheet = []

    confidence_level_id_checksheet = []

    output_folder = Path(IDENTIFIER_OUTPUT_FOLDER)

    if roi_id_checksheet.is_file():    
        for roi in loaded_roi_id_checksheet:
            key, x1, y1, x2, y2 = roi["key"], roi["x1"], roi["y1"], roi["x2"], roi["y2"]
            cropped = final_image[y1:y2, x1:x2]

            output_path = os.path.join(output_folder, f"crop_id_check.jpg")
            cv2.imwrite(output_path, cropped)

            # Check if the cropped region is empty
            is_filled = np.any(cropped == 0)  # Cek apakah ROI at least ada 1 pixel item
            if is_filled:
                cropped_processed = preprocessing_cropped_image_rfg(output_path)
                print("done")
                try:
                    prediction = model.predict(cropped_processed)
                except Exception as e:
                    print(f"error: {e}")
                text_id_checksheet.append({key:prediction[0]})

                # Ambil probabilitas prediksi
                probabilities = model.predict_proba(cropped_processed)
                # Confidence level: probabilitas tertinggi
                confidence = np.max(probabilities) 
                confidence_level_id_checksheet.append({"CT1": round(confidence * 100, ndigits=2)})

                print(f"Single Number: {prediction[0]}")
                print(f"Confidence Level for ROI: {confidence * 100:.2f}%")
                
            else:
                print("")
                text_id_checksheet.append({key:" "})
                confidence_level_id_checksheet.append({"CT1": 0.0})
                print("Empty cell")

    return text_id_checksheet, confidence_level_id_checksheet

def ocr_pink_digit_print_rfg(final_image, roi_printed_pink, loaded_roi_printed_pink, target_cell_cl):

    # Load model
    model = joblib.load(Path(RFG_DIGIT_PRINT_PINK))

    output_folder = Path(RFG_OUTPUT_FOLDER)

    cl_index = 0
    text_printed_pink = []
    confidence_level_printed_pink = []

    if roi_printed_pink.is_file():    
        for roi in loaded_roi_printed_pink:
            key, x1, y1, x2, y2 = roi["key"], roi["x1"], roi["y1"], roi["x2"], roi["y2"]
            cropped = final_image[y1:y2, x1:x2]

            output_path = os.path.join(output_folder, f"crop_2.jpg")
            cv2.imwrite(output_path, cropped)

            # Check if the cropped region is empty
            is_filled = np.any(cropped == 0)  # Cek apakah ROI isi
            if is_filled:
                cropped_processed = preprocessing_cropped_image_rfg(output_path)
                prediction = model.predict(cropped_processed)
                text_printed_pink.append({key:prediction[0]})

                # Ambil probabilitas prediksi
                probabilities = model.predict_proba(cropped_processed)
                # Confidence level: probabilitas tertinggi
                confidence = np.max(probabilities) 
                # Ambil key dari target_cell_cl berdasarkan urutan
                if cl_index < len(target_cell_cl):
                    cl_key = target_cell_cl[cl_index]
                    confidence_level_printed_pink.append({cl_key: round(confidence * 100, ndigits=2)})
                    cl_index += 1

                print(f"Single Number: {prediction[0]}")
                print(f"Confidence Level for ROI: {confidence * 100:.2f}%")

            else:
                print("")
                text_printed_pink.append("")
                if cl_index < len(target_cell_cl):
                    cl_key = target_cell_cl[cl_index]
                    confidence_level_printed_pink.append({cl_key: 0.0})
                    cl_index += 1
                print("Empty cell")

    print(text_printed_pink)
    print(confidence_level_printed_pink)

    return (confidence_level_printed_pink, text_printed_pink)

def ocr_pink_handw_rfg(final_image, roi_handw, loaded_roi_handwritten, target_cell_result_number, target_cell_cl_number, target_cell_result_text, target_cell_cl_text , ocr_prompts):
    print("TARGET CELL NUMBER RESULT NIH")
    print(target_cell_result_number)
    print(target_cell_cl_number)
    
    text_digit_handw_pink = []
    confidence_level_digit_handw_pink = []

    cl_index_number = 0
    digit_index = 0
    cl_index_text = 0
    letter_index = 0
    text_letter_handw_pink = []
    confidence_level_letter_handw_pink = []

    if roi_handw.is_file():
        for i, roi in enumerate(loaded_roi_handwritten): 
            category, type, x1, y1, x2, y2 = roi["category"], roi["type"], roi["x1"], roi["y1"], roi["x2"], roi["y2"]
            cropped = final_image[y1:y2, x1:x2]

            output_dir = r"/home/seizou-admin/PROJ/SHAREFOLDER/result/image"
            file_name = f"gambar_{i}.jpg"
            output_path = os.path.join(output_dir, file_name)

            cv2.imwrite(output_path, cropped)

            print(f"Processing ROI with category '{category}'")
            result, validation, validity_array = recognize_text(cropped, category, ocr_prompts)

            print(f"Extracted OCR: {result}, Validation: {validation}, Validity Array: {validity_array}")

            if type == "number":
                for i, val in enumerate(result):
                    if digit_index < len(target_cell_result_number):
                        key = target_cell_result_number[digit_index]
                        text_digit_handw_pink.append({key: val})
                        
                        confidence = validity_array[i] if i < len(validity_array) else 0.0
                        if cl_index_number < len(target_cell_cl_number):
                            cl_key = target_cell_cl_number[cl_index_number]
                            confidence_level_digit_handw_pink.append({cl_key: round(confidence * 100, ndigits=2)})
                            cl_index_number += 1

                        digit_index += 1

            elif type == "text":
                for i, val in enumerate(result):
                    if letter_index < len(target_cell_result_text):
                        key = target_cell_result_text[letter_index]
                        text_letter_handw_pink.append({key: val})
                        
                        confidence = validity_array[i] if i < len(validity_array) else 0.0
                        if cl_index_text < len(target_cell_cl_text):
                            cl_key = target_cell_cl_text[cl_index_text]
                            confidence_level_letter_handw_pink.append({cl_key: round(confidence * 100, ndigits=2)})
                            cl_index_text += 1

                        letter_index += 1

            else:
                print(result)
                for i, val in enumerate(result):
                    if i == 3 or i == 5:
                        if isinstance(result[3], int) or isinstance(result[5], int):
                            val = result[i]
                            if digit_index < len(target_cell_result_number):
                                key = target_cell_result_number[digit_index]
                                text_digit_handw_pink.append({key: val})

                                confidence = validity_array[i] if i < len(validity_array) else 0.0
                                if cl_index_number < len(target_cell_cl_number):
                                    cl_key = target_cell_cl_number[cl_index_number]
                                    confidence_level_digit_handw_pink.append({cl_key: round(confidence * 100, ndigits=2)})
                                    cl_index_number += 1
                                digit_index += 1
                                    
                        # Jika hasil OCR sudah habis tetapi masih ada target sel
                        elif result[i] == "#":
                            if digit_index < len(target_cell_result_number):
                                key = target_cell_result_number[digit_index]
                                text_digit_handw_pink.append({key: " "})
                                
                                confidence = validity_array[i] if i < len(validity_array) else 0.0
                                if cl_index_number < len(target_cell_cl_number):
                                    cl_key = target_cell_cl_number[cl_index_number]
                                    confidence_level_digit_handw_pink.append({cl_key: 0.0}) # Atau nilai default lain seperti 0.0
                                    cl_index_number += 1
                                digit_index += 1
                    
                    else:
                        if isinstance(result[i], str):
                            val = result[i]
                            # Pastikan kita tidak melebihi batas hasil OCR
                            if letter_index < len(target_cell_result_text):
                                key = target_cell_result_text[letter_index]
                                text_letter_handw_pink.append({key: val})

                                confidence = validity_array[i] if i < len(validity_array) else 0.0
                                if cl_index_text < len(target_cell_cl_text):
                                    cl_key = target_cell_cl_text[cl_index_text]
                                    confidence_level_letter_handw_pink.append({cl_key: round(confidence * 100, ndigits=2)})
                                    cl_index_text += 1
                                letter_index += 1
                                    
                            # # Jika hasil OCR sudah habis tetapi masih ada target sel
                            # elif letter_index < len(target_cell_result_text):
                            #     key = target_cell_result_text[letter_index]
                            #     text_letter_handw_pink.append({key: " "})
                            #     # Anda mungkin juga ingin menambahkan nilai default untuk confidence level di sini
                            #     if cl_index_text < len(target_cell_cl_text):
                            #         cl_key = target_cell_cl_text[cl_index_text]
                            #         confidence_level_letter_handw_pink.append({cl_key: 0.0}) # Atau nilai default lain seperti 0.0
                            #         cl_index_text += 1
                            #     letter_index += 1

    print(text_digit_handw_pink)
    print(len(text_digit_handw_pink))
    print(confidence_level_digit_handw_pink)
    print(len(confidence_level_digit_handw_pink))
    print(len(target_cell_result_number))

    print(text_letter_handw_pink)
    print(len(text_letter_handw_pink))
    print(confidence_level_letter_handw_pink)
    print(len(confidence_level_letter_handw_pink))
    print(len(target_cell_result_text))

    print(target_cell_cl_text)

    return(confidence_level_digit_handw_pink, text_digit_handw_pink, confidence_level_letter_handw_pink, text_letter_handw_pink)

def ocr_purple(
    final_image, 
    processed_black, 
    roi_purple, 
    loaded_roi_purple,
    tc_cl_purple
):
    text_purple = []
    confidence_level_purple = []
    processed_target_cells = []
    processed_confidence_levels = []
    loaded_new_roi = []

    if roi_purple.is_file():    
        for i, roi in enumerate(loaded_roi_purple):
            key, x1, y1, x2, y2 = roi["key"], roi["x1"], roi["y1"], roi["x2"], roi["y2"]
            black_cell = processed_black[y1:y2, x1:x2]
            black_cell[black_cell < 100] = 0
            
            height, width = black_cell.shape
            total_pixel = width * height
            
            count_black = np.sum(black_cell != 255)
            ratio_black = count_black / total_pixel * 100
            
            if ratio_black > 90:
                #skip roi skrng ke roi berikutny, skip target cell
                continue

            processed_target_cells.append(key) # nyimpen key result
            # Menambahkan ke loaded_new_roi hanya jika key ada di processed_target_cells
            if key in processed_target_cells:
                loaded_new_roi.append(roi)  # Menyimpan dictionary yang sesuai ke list
            processed_confidence_levels.append(tc_cl_purple[i]) # nyimpen key cl

        print()
        print(processed_target_cells)
        print(len(processed_target_cells))
        print(processed_confidence_levels)
        print(len(processed_confidence_levels))
        print(loaded_new_roi)
        print(len(loaded_new_roi))

        for i, roi in enumerate(loaded_new_roi):
            key, x1, y1, x2, y2 = roi["key"], roi["x1"], roi["y1"], roi["x2"], roi["y2"]
            cropped = final_image[y1:y2, x1:x2]

            height, width = cropped.shape
            total_pixel = width * height
            
            count_black = np.sum(cropped != 255)
            ratio_black = count_black / total_pixel * 100
            
            if ratio_black > 0.5:
                text_purple.append({processed_target_cells[i]:"O"})
                confidence_level_purple.append({processed_confidence_levels[i]:100})
            else:
                print("Empty ROI detected (based on threshold), processing anyway:")
                text_purple.append({processed_target_cells[i]:"X"})
                confidence_level_purple.append({processed_confidence_levels[i]:1})

        print()
        print(text_purple)
        print(len(text_purple))
        print(confidence_level_purple)
        print(len(confidence_level_purple))

    return (confidence_level_purple, text_purple)