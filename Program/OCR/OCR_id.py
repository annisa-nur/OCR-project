#!/usr/bin/env python
# coding: utf-8
import os
import cv2
import numpy as np
import joblib
from path_config import *
from pathlib import Path
from PREPROCESSING.preprocessing_cropped_rfg import *
from PREPROCESSING.preprocessing_printed import *

def ocr_id_scan(
    final_image,
    roi_id_scan,
    loaded_roi_id_scan      
):
    text_id_scan = []
    output_folder = Path(IDENTIFIER_OUTPUT_FOLDER)

    if roi_id_scan.is_file():    
        for roi in loaded_roi_id_scan:
            x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
            cropped = final_image[y1:y2, x1:x2]
            cropped[cropped >= 250] = 255

            output_path = os.path.join(output_folder, f"crop_id_scan.jpg")
            cv2.imwrite(output_path, cropped)

            height, width = cropped.shape
            total_pixel = height * width
            pixel_black = np.count_nonzero(cropped != 255)
            ratio_black = pixel_black / total_pixel * 100
            
            if ratio_black > 10:
                text_id_scan.append(False)
                continue
            else:
                text_id_scan.append(True)
                print(f"Process continue...., text: {text_id_scan[-1]}")

    return text_id_scan, cropped
    
