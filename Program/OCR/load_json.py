#!/usr/bin/env python
# coding: utf-8
import json
from pathlib import Path
import logging
from path_config import *

# Get the same logger instance as the main program
logger = logging.getLogger('__main__')

def load_json_scan():
    print("Loading json scan status...")
    logger.info("Loading json scan status...")  

    try:
        roi_id_scan = SCAN_ID_PATH

        with open(roi_id_scan, 'r') as f:
            scan_data = json.load(f)

        loaded_scan_checksheet = []
        for field_key, field_val in scan_data.items():
            loaded_scan_checksheet.append({
                "key": field_key,
                "x1": field_val["x1"],
                "y1": field_val["y1"],
                "x2": field_val["x2"],
                "y2": field_val["y2"]
            })
        print("Scan status have been loaded successfully.")
        logger.info("Scan status have been loaded successfully.")

    except Exception as e:
        print(f"Error loading Scan status: {e}")
        logger.error(f"Error loading Scan status: {e}")
    
    return roi_id_scan, loaded_scan_checksheet

def load_json_identifier():
    print("Loading json ROI identifier...")
    logger.info("Loading json ROI identifier...")  

    try:
        # initialize path for identifiers json
        roi_id_checksheet = CHECKSHEET_ID_PATH

        # load identifiers data as dictionary
        with open(roi_id_checksheet, 'r') as f:
            id_data = json.load(f)
        
        loaded_id_checksheet= []
        for field_key, field_val in id_data.items():
            loaded_id_checksheet.append({
                "key": field_key,
                "x1": field_val["x1"],
                "y1": field_val["y1"],
                "x2": field_val["x2"],
                "y2": field_val["y2"]
            })

        print("Checksheet identifiers have been loaded successfully.")
        logger.info("Checksheet identifiers have been loaded successfully.")

    except Exception as e:
        print(f"Error loading checksheet identifiers: {e}")
        logger.error(f"Error loading checksheet identifiers: {e}")

    return roi_id_checksheet, loaded_id_checksheet

def load_json_pink(roi_print, roi_handw):
    print("Loading json ROI pink...")
    logger.info("Loading json ROI pink...")
    try:
        # initialize path for roi pink json
        roi_print = Path(roi_print)
        roi_handw = Path(roi_handw)
        
        # load ROI print data as dictionary
        if roi_print.exists():
            with open(roi_print, 'r') as f:
                data_print = json.load(f)
            logger.info(f"File successfully loaded: {roi_print}")
        else:
            print(f"File can't be found: {roi_print}")
            logger.error(f"File can't be found: {roi_print}")
            data_print = {}

        loaded_roi_print = []
        for field_key, field_val in data_print.items():
            for field_name, details in field_val.items():
                loaded_roi_print.append({
                        "name": field_key,
                        "key": field_name,
                        "x1": details["x1"],
                        "y1": details["y1"],
                        "x2": details["x2"],
                        "y2": details["y2"]
                })
        print("ROI print:", loaded_roi_print, "\n")

        # load ROI handw data as dictionary
        if roi_handw.exists():
            with open(roi_handw, 'r') as f:
                data_handw = json.load(f)
        else:
            print(f"File handw can't be found: {roi_handw}")
            logger.error(f"File handw can't be found: {roi_handw}")
            data_handw = {}

        loaded_roi_cdnn = []
        loaded_roi_gemini = []
        for field_key, field_val in data_handw.items():
            if "dl" in field_val:
                for cell_key, cell_val in field_val["dl"].items():
                    loaded_roi_cdnn.append({
                        "name": field_key,
                        "key": cell_key,
                        "x1": cell_val["x1"],
                        "y1": cell_val["y1"],
                        "x2": cell_val["x2"],
                        "y2": cell_val["y2"],
                        "type": cell_val["type"]
                    })
            if "gemini" in field_val:
                gemini_ai = field_val["gemini"]
                # for cell_key, cell_val in field_val["gemini"].items():
                loaded_roi_gemini.append({
                        "x1": gemini_ai["x1"],
                        "y1": gemini_ai["y1"],
                        "x2": gemini_ai["x2"],
                        "y2": gemini_ai["y2"],
                        "type": gemini_ai["type"],
                        "category": gemini_ai["category"]
                })
        
        print("ROI cdnn:", loaded_roi_cdnn, "\n")
        print("ROI gemini:", loaded_roi_gemini, "\n")
        logger.info("ROI pink have been loaded successfully.")
    
    except Exception as e:
        print(f"Error loading ROI pink: {e}")
        logger.error(f"Error loading ROI pink: {e}")

    return loaded_roi_print, loaded_roi_cdnn, loaded_roi_gemini

def load_json_purple(roi_purple):    
    print("Loading json ROI purple...")
    logger.info("Loading json ROI purple...")

    try:
        # initialize path for roi pink json
        roi_purple = Path(roi_purple)

        # load ROI purple data as dictionary
        if roi_purple.exists():
            with open(roi_purple, 'r') as f:
                data_purple = json.load(f)
        else:
            print(f"File handw can't be found: {roi_purple}")
            data_purple = {}

        loaded_roi_purple = []
        for field_key, field_val in data_purple.items():
            loaded_roi_purple.append({
                "key": field_key,
                "x1": field_val["x1"],
                "y1": field_val["y1"],
                "x2": field_val["x2"],
                "y2": field_val["y2"]
            })
        print("ROI purple:", loaded_roi_purple, "\n")
        print(len(loaded_roi_purple))
        logger.info("ROI purple have been loaded successfully.")

    except Exception as e:
        print(f"Error loading ROI purple: {e}")
        logger.error(f"Error loading ROI purple: {e}")

    return roi_purple, loaded_roi_purple

def load_json_target_cells(roi_print, roi_pink, roi_purple, loaded_roi_print, loaded_roi_cdnn, loaded_roi_purple):
    def col_to_index(col):
        col = col.upper()
        index = 0
        for char in col:
            index = index * 26 + (ord(char) - ord('A') + 1)
        return index

    def index_to_col(index):
        col = ""
        while index > 0:
            index -= 1
            col = chr(index % 26 + ord('A')) + col
            index //= 26
        return col

    def shift_cell(cell, shift=96):
        col_part = ''.join(filter(str.isalpha, cell))
        row_part = ''.join(filter(str.isdigit, cell))
        new_col_index = col_to_index(col_part) + shift
        new_col = index_to_col(new_col_index)
        return f"{new_col}{row_part}"

    try:
        # load roi data
        with open(roi_purple, 'r') as f:
            check_data = json.load(f)
        with open(roi_print, 'r') as f:
            print_data = json.load(f)
        with open(roi_pink, 'r') as f:
            handw_data = json.load(f)

        result_output = {
            "result_purple": [],
            "result_digit_print_pink": [],
            "result_digit_handw_pink": [],
            "result_letter_pink": []
        }
        for i, item in enumerate(loaded_roi_print):
            result_output["result_digit_print_pink"].append(item['key'])
        for item in loaded_roi_cdnn:
            if item["type"] == "number":
                result_output["result_digit_handw_pink"].append(item["key"])
            elif item["type"] == "text":
                result_output["result_letter_pink"].append(item["key"])
        for key in loaded_roi_purple:
            result_output["result_purple"].append(key)

        confidence_output = {
            "confidence_level_purple": [],
            "confidence_level_digit_print_pink": [],
            "confidence_level_digit_handw_pink": [],
            "confidence_level_letter_pink": []
        }

        for cell in check_data:
            shifted = shift_cell(cell)
            confidence_output["confidence_level_purple"].append(shifted)

        for cell in result_output["result_digit_print_pink"]:
            shifted = shift_cell(cell)
            confidence_output["confidence_level_digit_print_pink"].append(shifted)

        for section in handw_data.values():
            dl_section = section.get("dl", {})
            for cell, data in dl_section.items():
                shifted = shift_cell(cell)
                cell_type = data.get("type", "").lower()
                if cell_type == "number":
                    confidence_output["confidence_level_digit_handw_pink"].append(shifted)
                elif cell_type in ["text", "letter"]:
                    confidence_output["confidence_level_letter_pink"].append(shifted)
        
        print("Target cells have been loaded successfully.")
        logger.info("Target cells have been loaded successfully.")
        print("target cells")
        print(result_output["result_digit_handw_pink"],
        result_output["result_letter_pink"])

    except Exception as e:
        print(f"Error loading target cells: {e}")
        logger.error(f"Error loading target cels: {e}")

    return (
        result_output["result_digit_handw_pink"],
        result_output["result_letter_pink"],
        confidence_output["confidence_level_purple"],
        confidence_output["confidence_level_digit_print_pink"],
        confidence_output["confidence_level_digit_handw_pink"],
        confidence_output["confidence_level_letter_pink"]
    )


