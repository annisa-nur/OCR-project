import numpy as np
import os
import time
import cv2
from path_config import *
from PREPROCESSING.preprocessing_black_cell import *
from PREPROCESSING.preprocessing_printed import *
from PREPROCESSING.preprocessing_cropped_rfg import *
from PREPROCESSING.preprocessing_cropped_cnn import *
from PREPROCESSING.preprocessing_cropped_dnn import *
from PREPROCESSING.preprocessing_cropped_ann import *
from OCR_PROGRAM.load_json import *
from OCR_PROGRAM.OCR_cnn import *
from OCR_PROGRAM.OCR_rfg import *
from OCR_PROGRAM.OCR_rfg import ocr_pink_digit_print_rfg
from OCR_PROGRAM.OCR_dnn import *
from OCR_PROGRAM.OCR_ann import *
from OCR_PROGRAM.OCR_resnet18 import *
# from model_ensemble_old import *
from save_to_excel import *
from Result_analysis import *
from collections import defaultdict

# Convert list of dicts → dict (biar bisa akses by key langsung)
def list_of_dicts_to_dict(lod):
    if not lod:
        return {}
    result_dict = {}
    for d in lod:
        if d:  # Check if the dictionary is not empty
            key_list = list(d.keys())
            value_list = list(d.values())
            if key_list and value_list: # Check if keys and values exist
                result_dict[key_list[0]] = value_list[0]
    return result_dict

def get_checksheet_path(text_id_checksheet, path_type):
    variable_name = f"CHECKSHEET{text_id_checksheet}_{path_type}"
    
    checksheet_path = globals().get(variable_name)

    if checksheet_path is {}:
        raise ValueError(f"Variable {variable_name} not found!")
    
    return checksheet_path

def do_ocr(time_stamp, text_id_checksheet, final_idc, tc_result_idc, tc_result_id, list_of_var_id, image, processed_image, processed_black, ocr_prompts, image_path, excel_path, wb_target, model_cnn_status, model_dnn_status, model_ann_status, model_resnet_status, model_rfg_status, cropped):
    print(f"doing ocr for page {text_id_checksheet}...")
    processed_image_path = get_checksheet_path(text_id_checksheet, "PROCESSED_IMAGE")
    cv2.imwrite(processed_image_path, processed_image)
    directory, filename = os.path.split(image_path)  # Memisahkan direktori dan nama file
    new_filename = f"checksheet_{text_id_checksheet}_{time_stamp}.jpg"  # Menentukan nama baru
    new_path = os.path.join(directory, new_filename)  # Membuat path baru
    os.rename(image_path, new_path)

    output_folder = Path(IDENTIFIER_OUTPUT_FOLDER)
    if text_id_checksheet == 1:
        sheet_type = "工程QC（1）"
        output_path = os.path.join(output_folder, f"crop_id_scan1.jpg")
        cv2.imwrite(output_path, cropped)
    elif text_id_checksheet == 2:
        sheet_type = "工程QC（2）"
        output_path = os.path.join(output_folder, f"crop_id_scan2.jpg")
        cv2.imwrite(output_path, cropped)
    elif text_id_checksheet == 3:
        sheet_type = "工程QC（検品・梱包ー1）"
        output_path = os.path.join(output_folder, f"crop_id_scan3.jpg")
        cv2.imwrite(output_path, cropped)
    elif text_id_checksheet == 6:
        sheet_type = "CCP3"
        output_path = os.path.join(output_folder, f"crop_id_scan6.jpg")
        cv2.imwrite(output_path, cropped)
    elif text_id_checksheet == 7:
        sheet_type = "使用洗浄（前）" 
        output_path = os.path.join(output_folder, f"crop_id_scan7.jpg")
        cv2.imwrite(output_path, cropped)
    elif text_id_checksheet == 8:
        sheet_type = "使用洗浄（後）"
        output_path = os.path.join(output_folder, f"crop_id_scan8.jpg")
        cv2.imwrite(output_path, cropped)
    else:
        os.remove(new_path)
        checksheet_status = False
        return new_path #, checksheet_status

    roi_print = Path(get_checksheet_path(text_id_checksheet, "ROI_PRINT"))
    roi_handw = Path(get_checksheet_path(text_id_checksheet, "ROI_HANDW"))
    roi_purple = Path(get_checksheet_path(text_id_checksheet, "ROI_PURPLE"))

    loaded_roi_print, loaded_roi_handw, loaded_roi_gemini = load_json_pink(roi_print, roi_handw)
    roi_purple, loaded_roi_purple = load_json_purple(roi_purple)
    # load external target cells
    target_cell_result_number, target_cell_result_text, target_cell_cl_purple, target_cell_cl_print, target_cell_cl_number, target_cell_cl_text = load_json_target_cells(roi_print, roi_handw, roi_purple, loaded_roi_print, loaded_roi_handw, loaded_roi_purple)
        # Menggunakan status model untuk menentukan model mana yang aktif
    if model_cnn_status:
        proba_print_pink_cnn, tc_result_print_pink = ocr_pink_digit_print_cnn(processed_image, roi_print, loaded_roi_print, target_cell_cl_print)
        proba_digit_handw_pink_cnn, tc_digit_handw_pink, proba_letter_pink_cnn, tc_letter_handw_pink = ocr_pink_handw_cnn(processed_image, roi_handw, loaded_roi_handw, target_cell_cl_number, target_cell_cl_text)
    else:
        proba_print_pink_cnn, tc_result_print_pink = {}, {}
        proba_digit_handw_pink_cnn, tc_digit_handw_pink, proba_letter_pink_dnn, tc_letter_handw_pink = {}, {}, {}, {}

    if model_dnn_status:
        proba_print_pink_dnn = ocr_pink_digit_print_dnn(processed_image, roi_print, loaded_roi_print, target_cell_cl_print)
        proba_digit_handw_pink_dnn, proba_letter_pink_dnn = ocr_pink_handw_dnn(processed_image, roi_handw, loaded_roi_handw, target_cell_cl_number, target_cell_cl_text)
    else:
        proba_print_pink_dnn = {}
        proba_digit_handw_pink_dnn, proba_letter_pink_dnn = {}, {}

    if model_ann_status:
        proba_print_pink_ann = ocr_pink_digit_print_ann(processed_image, roi_print, loaded_roi_print, target_cell_cl_print, text_id_checksheet)
        proba_digit_handw_pink_ann, proba_letter_pink_ann = ocr_pink_handw_ann(processed_image, roi_handw, loaded_roi_handw, target_cell_cl_number, target_cell_cl_text, text_id_checksheet)
    else:
        proba_print_pink_ann = {}
        proba_digit_handw_pink_ann, proba_letter_pink_ann = {}, {}

    if model_resnet_status:
        proba_digit_handw_pink_resnet, proba_letter_pink_resnet = ocr_pink_handw_resnet(processed_image, roi_handw, loaded_roi_handw, target_cell_cl_number, target_cell_cl_text)
    else:
        proba_digit_handw_pink_resnet, proba_letter_pink_resnet = {}, {}

    if model_rfg_status:
        proba_print_pink_rfg = ocr_pink_digit_print_rfg(processed_image, roi_print, loaded_roi_print, target_cell_cl_print)
        proba_digit_handw_pink_rfg, proba_letter_pink_rfg = ocr_pink_handw_rfg(processed_image, roi_handw, loaded_roi_gemini, target_cell_result_number, target_cell_cl_number, target_cell_result_text, target_cell_cl_text, ocr_prompts)
    else:
        proba_print_pink_rfg = {}
        proba_digit_handw_pink_rfg, proba_letter_pink_rfg = {}, {}


    final_handw_digit_result, final_confidence_lvl_handw_digit, list_of_var_digit = ensemble_model_digit_handw(
                               proba_digit_handw_pink_cnn, tc_digit_handw_pink,
                               proba_digit_handw_pink_dnn,
                               proba_digit_handw_pink_ann,
                               proba_digit_handw_pink_resnet,
                               proba_digit_handw_pink_rfg)
    final_handw_letter_result, final_confidence_lvl_handw_letter, list_of_var_letter = ensemble_model_letter_handw(
                               proba_letter_pink_cnn,
                               tc_letter_handw_pink,
                               proba_letter_pink_dnn,
                               proba_letter_pink_ann,
                               proba_letter_pink_resnet,
                               proba_letter_pink_rfg)
    final_print_digit_result, final_confidence_lvl_print_digit, list_of_var_print = ensemble_model_digit_print(  
                               proba_print_pink_cnn,
                               tc_result_print_pink,
                               proba_print_pink_dnn,
                               proba_print_pink_ann,
                               proba_print_pink_rfg)
    final_print_digit_result.insert(0, final_idc)
    final_confidence_lvl_print_digit.insert(0, tc_result_idc)


    id_analysis_path = result_id_analysis(list_of_var_id, tc_result_id)
    number_analysis_path = result_number_analysis(list_of_var_digit, tc_digit_handw_pink)
    letter_analysis_path = result_letter_analysis(list_of_var_letter, tc_letter_handw_pink)
    print_analysis_path = result_print_analysis(list_of_var_print, tc_result_print_pink)
    join_all_model_predictions(id_analysis_path, letter_analysis_path, number_analysis_path, print_analysis_path)
    save_results_ensemble_to_txt(final_print_digit_result,
                                final_confidence_lvl_print_digit,
                                final_handw_digit_result,
                                final_confidence_lvl_handw_digit,
                                final_handw_letter_result,
                                final_confidence_lvl_handw_letter)

    # Mencetak hasil dari 6 variabel
    print("Final Print Digit Result:")
    print(final_print_digit_result)
    print("Final Confidence Level for Print Digit:")
    print(final_confidence_lvl_print_digit)

    print("\nFinal Handwritten Digit Result:")
    print(final_handw_digit_result)
    print("Final Confidence Level for Handwritten Digit:")
    print(final_confidence_lvl_handw_digit)

    print("\nFinal Handwritten Letter Result:")
    print(final_handw_letter_result)
    print("Final Confidence Level for Handwritten Letter:")
    print(final_confidence_lvl_handw_letter)

    # tc_result_print, tc_result_digit_handw_pink, tc_result_letter_pink, tc_result_purple, tc_confidence_level_print, tc_confidence_level_digit_handw_pink, tc_confidence_level_letter_pink, tc_confidence_level_purple = load_json_target_cells(roi_print, roi_pink, roi_purple, loaded_roi_print, loaded_roi_cdnn, loaded_roi_gemini, loaded_roi_purple)
    confidence_level_purple, text_purple = ocr_purple(processed_image, processed_black, roi_purple, loaded_roi_purple, target_cell_cl_purple)

    print("preprocessing image for saving")
    start_preprocess = time.time()
    preprocessed_save_image = preprocessing_image_saving(image, image_path)
    preprocess_time = time.time() - start_preprocess
    print(f"Waktu preprocess: {preprocess_time:.2f} detik")
        
    print("compressing image")
    new_filename = f"checksheet_3_{time_stamp}.jpeg"
    compressed_dir = COMPRESSED_IMAGE_PATH
    save_image_path = compress_image(preprocessed_save_image, compressed_dir, new_filename)
    save_image = cv2.imread(save_image_path)

    save_result_python(sheet_type, excel_path, save_image,
                final_print_digit_result, final_handw_digit_result, final_handw_letter_result,
                final_confidence_lvl_print_digit, final_confidence_lvl_handw_digit, final_confidence_lvl_handw_letter,
                confidence_level_purple, text_purple,
                wb_target)

    print("well done")
    # status = True
    return new_path #, checksheet_status

def ensemble_model_digit_print(
    proba_print_pink_cnn,
    tc_result_print_pink,
    proba_print_pink_dnn,
    proba_print_pink_ann,
    proba_print_pink_rfg
):
    list_pool = []
    result = []
    class_label = []

    list_of_vars = [
        proba_print_pink_cnn,
        proba_print_pink_dnn,
        proba_print_pink_ann,
        proba_print_pink_rfg
    ]

    for var in list_of_vars:
        if isinstance(var, list) and var:
            list_pool.append(var)
    
    print(list_pool)
    # Menyimpan hasil rata-rata terbesar

    # Ambil semua key yang ada di list of dictionaries
    keys = []
    for sublist in list_pool:
        for dic in sublist:
            keys.extend(dic.keys())  # Menambahkan key ke dalam set

    final_keys = []
    [final_keys.append(key) for key in keys if key not in final_keys]
    print(f"final keys {final_keys}")

    # Untuk setiap key yang ditemukan, hitung rata-rata terbesar
    for key in final_keys:
        values_for_key = []

        # Iterasi setiap list di list_of_lists
        for sublist in list_pool:
            # Ambil nilai untuk key yang sesuai dalam sublist
            for dic in sublist:
                if key in dic:
                    values_for_key.append(dic[key])

        values_for_key_transposed = np.transpose(values_for_key)

        # Menghitung rata-rata terbesar untuk setiap key
        avg_values = np.mean(values_for_key_transposed, axis=1)  # Ambil nilai terbesar dari setiap posisi
        max_avg_value = np.max(avg_values)  # Rata-rata nilai terbesar
        index_of_max_avg = np.argmax(avg_values)

        # Tambahkan hasil dalam format dictionary
        result.append({key: max_avg_value * 100})
        if max_avg_value != 0.0:
            class_label.append(index_of_max_avg)
        else:
            class_label.append("")

    # Menampilkan hasil akhir
    print(result)
    # Mendapatkan label kelas
    final_text = []

    for label, tc in zip(class_label, tc_result_print_pink):
        final_text.append({tc: label})
    
    print("List Final Result:", final_text)

    return final_text, result, list_of_vars

def ensemble_model_digit_handw(
    proba_digit_handw_pink_cnn,
    tc_digit_handw_pink,
    proba_digit_handw_pink_dnn,
    proba_digit_handw_pink_ann,
    proba_digit_handw_pink_resnet,
    proba_digit_handw_pink_rfg
):
    
    print(f"{proba_digit_handw_pink_cnn} /n")
    print(f"{proba_digit_handw_pink_dnn} /n")
    print(f"{proba_digit_handw_pink_ann} /n")
    print(f"{proba_digit_handw_pink_resnet} /n")
    print(f"{proba_digit_handw_pink_rfg}")

    list_pool = []
    result = []
    class_label = []

    list_of_vars = [
        proba_digit_handw_pink_cnn,
        proba_digit_handw_pink_dnn,
        proba_digit_handw_pink_ann,
        proba_digit_handw_pink_resnet,
        proba_digit_handw_pink_rfg
    ]

    for var in list_of_vars:
        if isinstance(var, list) and var:
            list_pool.append(var)
    
    print(list_pool)
    # Menyimpan hasil rata-rata terbesar

    # Ambil semua key yang ada di list of dictionaries
    keys = []
    for sublist in list_pool:
        for dic in sublist:
            keys.extend(dic.keys())  # Menambahkan key ke dalam set

    final_keys = []
    [final_keys.append(key) for key in keys if key not in final_keys]
    print(f"final keys {final_keys}")

    # Untuk setiap key yang ditemukan, hitung rata-rata terbesar
    for key in final_keys:
        values_for_key = []

        # Iterasi setiap list di list_of_lists
        for sublist in list_pool:
            # Ambil nilai untuk key yang sesuai dalam sublist
            for dic in sublist:
                if key in dic:
                    values_for_key.append(dic[key])

        values_for_key_transposed = np.transpose(values_for_key)

        # Menghitung rata-rata terbesar untuk setiap key
        avg_values = np.mean(values_for_key_transposed, axis=1)  # Ambil nilai terbesar dari setiap posisi
        max_avg_value = np.max(avg_values)  # Rata-rata nilai terbesar
        index_of_max_avg = np.argmax(avg_values)

        # Tambahkan hasil dalam format dictionary
        result.append({key: max_avg_value * 100})
        if max_avg_value != 0.0:
            class_label.append(index_of_max_avg)
        else:
            class_label.append("")

    # Menampilkan hasil akhir
    print(result)
    # Mendapatkan label kelas
    final_text = []

    for label, tc in zip(class_label, tc_digit_handw_pink):
        final_text.append({tc: label})
    
    print("List Final Result:", final_text)

    return final_text, result, list_of_vars

def ensemble_model_letter_handw(
    proba_letter_pink_cnn,
    tc_letter_handw_pink,
    proba_letter_pink_dnn,
    proba_letter_pink_ann,
    proba_letter_pink_resnet,
    proba_letter_pink_rfg
):
    list_pool = []
    result = []
    class_label = []
    alphabet = string.ascii_uppercase

    list_of_vars = [
        proba_letter_pink_cnn,
        proba_letter_pink_dnn,
        proba_letter_pink_ann,
        proba_letter_pink_resnet,
        proba_letter_pink_rfg
    ]

    for var in list_of_vars:
        if isinstance(var, list) and var:
            list_pool.append(var)
    
    print(list_pool)

    # Ambil semua key yang ada di list of dictionaries
    keys = []
    for sublist in list_pool:
        for dic in sublist:
            keys.extend(dic.keys())  # Menambahkan key ke dalam set

    final_keys = []
    [final_keys.append(key) for key in keys if key not in final_keys]
    print(f"final keys {final_keys}")

    # Untuk setiap key yang ditemukan, hitung rata-rata terbesar
    for key in final_keys:
        values_for_key = []

        # Iterasi setiap list di list_of_lists
        for sublist in list_pool:
            # Ambil nilai untuk key yang sesuai dalam sublist
            for dic in sublist:
                if key in dic:
                    values_for_key.append(dic[key])

        values_for_key_transposed = np.transpose(values_for_key)

        # Menghitung rata-rata terbesar untuk setiap key
        avg_values = np.mean(values_for_key_transposed, axis=1)  # Ambil nilai terbesar dari setiap posisi
        max_avg_value = np.max(avg_values)  # Rata-rata nilai terbesar
        index_of_max_avg = np.argmax(avg_values)

        # Tambahkan hasil dalam format dictionary
        result.append({key: max_avg_value * 100})
        if max_avg_value != 0.0:
            class_label.append(alphabet[index_of_max_avg])
        else:
            class_label.append("")

    # Menampilkan hasil akhir
    print(result)
    # Mendapatkan label kelas
    final_text = []

    for label, tc in zip(class_label, tc_letter_handw_pink):
        final_text.append({tc: label})
    
    print("List Final Result:", final_text)

    return final_text, result, list_of_vars