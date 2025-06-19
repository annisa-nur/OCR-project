#!/usr/bin/env python
# coding: utf-8
import base64
import cv2
import json
import google.generativeai as genai
import re
from path_config import *
from datetime import datetime
from google.generativeai.types import GenerationConfig

# Konfigurasi API Gemini
API_KEY = "AIzaSyC8HvroEh-1hjUyPd3cPN7FYsttHgq7GpA"
genai.configure(api_key=API_KEY, transport='rest')

def load_prompts(file_path=None):
    if file_path is None:
        file_path = RFG_PROMPTS_PATH  # Baru ambil dari path di sini
    
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# Fungsi untuk mengubah gambar numpy ke base64
def numpy_to_base64(image_np):
    _, buffer = cv2.imencode('.jpg', image_np)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

# Load prompt from JSON file
# ocr_prompts = load_prompts()

response_schema = {
    "type" : "object",
    "properties" : {
        "ocr_result":{
            "type" : "string",
            "description" : "Extracted text from the image",
        },
    },
    "required" : ["ocr_result"],
}

# Fungsi untuk memproses gambar dengan Generative AI
def process_image_with_gemini(image, prompt):
    base64_image = numpy_to_base64(image)
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    response = model.generate_content(
        [
            {"mime_type": "image/jpeg", "data": base64_image},  
            prompt
        ],
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            temperature=0,
            response_schema=response_schema
        )
    )
    # Pastikan response.text tidak kosong
    if not response.text:
        print("Error: Empty response from Gemini API")
        return None

    # Extract result
    text = json.loads(response.text)
    ocr_text = text.get('ocr_result', 'No text found')
    return ocr_text

# Prompt untuk berbagai tipe OCR
# Response & Prompt
def numpy_to_base64(image_np):
    _, buffer = cv2.imencode('.jpg', image_np)  # Convert image (NumPy array) to JPEG format
    base64_str = base64.b64encode(buffer).decode('utf-8')  # Encode as Base64 string
    return base64_str
    
# Validation
import re
from datetime import datetime

def validate_main_filler(ocr_output):
    # Expected length
    expected_lengths = 6
    
    # Ubah huruf kecil menjadi huruf besar
    ocr_output = ocr_output.upper()
    
    # Bersihkan hasil OCR (hanya huruf besar dan spasi)
    cleaned_output = re.sub(r"[^A-Z ]", "", ocr_output)
    
    # Pastikan panjang hasil sesuai format
    if len(cleaned_output) > expected_lengths:
        cleaned_output = cleaned_output[:expected_lengths]
    elif len(cleaned_output) < expected_lengths:
        cleaned_output = cleaned_output.ljust(expected_lengths, ' ')

    # Validasi karakter satu per satu
    character_validity = [char.isalpha() or char == " " for char in cleaned_output]

    # Pastikan semua karakter valid
    is_valid = all(character_validity)

    return is_valid, list(cleaned_output), character_validity

def validate_implementer(ocr_output):
    # Hapus karakter selain huruf besar dan spasi
    cleaned_output = re.sub(r"[^A-Z ]", "", ocr_output)
    # Pastikan panjang hasil sesuai salah satu format
    if len(cleaned_output) > 3:
        cleaned_output = cleaned_output[:3]
    elif len(cleaned_output) < 3:
        cleaned_output = cleaned_output.ljust(3, ' ')

    # Pola yang diperbolehkan: "LLL" atau "LL "
    if not re.match(r"^[A-Z]{3}$|^[A-Z]{2} $", cleaned_output):
        return False, list(cleaned_output), [False] * len(cleaned_output)
    
    is_valid = True
    character_validity = [True] * len(cleaned_output)

    # Semua karakter valid jika lolos aturan di atas
    return is_valid, list(cleaned_output), character_validity

def validate_date(ocr_output):
    # Hapus karakter selain angka
    cleaned_output = re.sub(r"\D", "", ocr_output)

    # Jika setelah pembersihan kosong, set ke "00000000"
    if not cleaned_output:
        return False, [0] * 8, [False] * 8

    # Pastikan panjangnya 8 digit, jika kurang lakukan padding, jika lebih potong
    if len(cleaned_output) > 8:
        cleaned_output = cleaned_output[:8]
    elif len(cleaned_output) < 8:
        cleaned_output = cleaned_output.ljust(8, '0')  # Tambah '0' di kanan

    # Pastikan semua karakter adalah digit
    if not cleaned_output.isdigit():
        return False, [0] * 8, [False] * 8

    # Konversi cleaned_output menjadi list integer
    cleaned_list = [int(digit) for digit in cleaned_output]

    # Ambil bagian tahun, bulan, dan hari
    year = int(cleaned_output[:4])
    month = int(cleaned_output[4:6])
    day = int(cleaned_output[6:])

    # Validasi bulan (01-12)
    if month < 1 or month > 12:
        return False, cleaned_list, [False] * 8

    # Jumlah hari dalam tiap bulan
    days_in_month = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
    }

    # Periksa tahun kabisat untuk Februari
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        days_in_month[2] = 29  # Tahun kabisat

    # Validasi hari dalam bulan
    if day < 1 or day > days_in_month[month]:
        return False, cleaned_list, [False] * 8

    # **Validasi terhadap tanggal hari ini**
    today = int(datetime.today().strftime("%Y%m%d"))  # Ambil tanggal hari ini sebagai integer
    if int(cleaned_output) > today:
        return False, cleaned_list, [False] * 8  # Jika tanggal lebih besar dari hari ini, invalid

    # Jika semua validasi lolos
    return True, cleaned_list, [True] * 8

def validate_product_tank(ocr_output):
    # Bersihkan output, hanya angka dan spasi
    cleaned_output = re.sub(r"[^\d ]", "", ocr_output)

    # Ambil hanya digit angka, abaikan spasi
    digits_only = [char for char in cleaned_output if char.isdigit()]

    # Jika jumlah angka kurang dari 4, dianggap tidak valid
    if len(digits_only) < 4:
        # Tambah 0 agar panjang jadi 4
        padded_digits = digits_only + ['0'] * (4 - len(digits_only))
        character_validity = [False] * 4  # Karena input aslinya gak cukup
        return False, [int(d) for d in padded_digits], character_validity

    # Kalau lebih dari 4 angka, ambil hanya 4 pertama
    digits_only = digits_only[:4]
    
    # Semua dianggap valid karena ada minimal 4 angka
    character_validity = [True] * 4

    return True, [int(d) for d in digits_only], character_validity

def validate_volume(ocr_output):
    # Hapus karakter selain digit dan spasi
    cleaned_output = re.sub(r"[^0-9 ]", "", ocr_output)

    # Hapus spasi di awal angka
    cleaned_output = cleaned_output.lstrip()

    # Hapus semua spasi di dalam angka, tetapi pertahankan urutannya
    digits_only = re.sub(r"\s+", "", cleaned_output)

    # Jika kurang dari 3 digit, return tidak valid tetapi tetap format angka di kiri
    if len(digits_only) < 3:
        digits_only = digits_only.ljust(5, '0')  # Gunakan '0' sebagai padding
        return False, [int(d) for d in digits_only], [False] * 5

    # Jika lebih dari 5 digit, ambil hanya 5 angka pertama
    if len(digits_only) > 5:
        digits_only = digits_only[:5]

    # Tambahkan nol di kanan jika kurang dari 5 karakter
    cleaned_output = digits_only.ljust(5, '0')

    # Validasi: Harus memiliki minimal 3 angka
    is_valid = len(digits_only) >= 3
    character_validity = [True] * 5  # Semua angka dianggap valid setelah pemrosesan

    return is_valid, [int(d) for d in cleaned_output], character_validity

def validate_volume_ccp(ocr_output):
    # Hapus karakter selain digit dan spasi
    cleaned_output = re.sub(r"[^0-9 ]", "", ocr_output).strip()

    # Hapus semua spasi di antara angka
    cleaned_output = re.sub(r"\s+", "", cleaned_output)

    # Pastikan panjang hasil sesuai format (2 karakter)
    if len(cleaned_output) > 2:
        cleaned_output = cleaned_output[:2]  # Ambil hanya 2 karakter pertama
    else:
        cleaned_output = cleaned_output.ljust(2, '0')  # Tambahkan '0' jika kurang

    # Valid formats: "NN", "N ", "  " (double space)
    is_valid = bool(re.match(r"^\d{2}$|^\d0$|^00$", cleaned_output))
    character_validity = [is_valid] * 2  # Dua karakter, jadi validitasnya dua elemen

    return is_valid, [int(d) for d in cleaned_output], character_validity

def validate_lot1(ocr_output):
    # Konversi ke huruf besar
    ocr_output = ocr_output.upper()

    # Split by commas to get individual rows
    rows = ocr_output.split(",")
    print("Split Rows:", rows)

    parsed_rows_final = []
    all_validity = []
    is_valid = True

    for i in range(3):
        row = ""
        if i < len(rows):
            row = rows[i].strip()

        # Hapus semua spasi dalam satu row
        row = row.replace(' ', '')

        # Pastikan panjang 6 karakter
        if len(row) > 6:
            row = row[:6]
        else:
            row = row.ljust(6, ' ')

        row_list = list(row)
        parsed_row = []

        for j, char in enumerate(row_list):
            if (j < 3 and char.isalpha()):
                parsed_row.append(char)
                all_validity.append(True)
            elif (j == 3 and char.isdigit()):
                parsed_row.append(int(char))
                all_validity.append(True)
            elif (j == 4 and char.isalpha()):
                parsed_row.append(char)
                all_validity.append(True)
            elif (j == 5 and char.isdigit()):
                parsed_row.append(int(char))
                all_validity.append(True)
            else:
                parsed_row.append(char)
                all_validity.append(False)
                is_valid = False

        parsed_rows_final.extend(parsed_row)

    print("Parsed Rows:", parsed_rows_final)
    print("Validity Flags:", all_validity)

    return is_valid, parsed_rows_final, all_validity

def validate_lot2(ocr_output):
    # Konversi ke huruf besar
    ocr_output = ocr_output.upper()

    # Split by commas to get individual rows
    rows = ocr_output.split(",")
    print("Split Rows:", rows)

    parsed_rows_final = []
    all_validity = []
    is_valid = True

    for i in range(4):
        row = ""
        if i < len(rows):
            row = rows[i].strip()

        # Hapus semua spasi dalam satu row
        row = row.replace(' ', '')

        # Pastikan panjang 6 karakter
        if len(row) > 6:
            row = row[:6]
        else:
            row = row.ljust(6, ' ')

        row_list = list(row)
        parsed_row = []

        for j, char in enumerate(row_list):
            if (j < 3 and char.isalpha()):
                parsed_row.append(char)
                all_validity.append(True)
            elif (j == 3 and char.isdigit()):
                parsed_row.append(int(char))
                all_validity.append(True)
            elif (j == 4 and char.isalpha()):
                parsed_row.append(char)
                all_validity.append(True)
            elif (j == 5 and char.isdigit()):
                parsed_row.append(int(char))
                all_validity.append(True)
            else:
                parsed_row.append(char)
                all_validity.append(False)
                is_valid = False

        parsed_rows_final.extend(parsed_row)

    print("Parsed Rows:", parsed_rows_final)
    print("Validity Flags:", all_validity)

    return is_valid, parsed_rows_final, all_validity

def validate_loss_1(ocr_output):
    print("Extracted OCR before cleaning:", ocr_output)  # Debugging

    # Hapus karakter selain digit, spasi, dan koma
    cleaned_rows = re.sub(r"[^\d,]", "", ocr_output)

    # Pastikan koma tidak berlebihan (hapus koma di awal/akhir)
    cleaned_rows = cleaned_rows.strip(",")

    # Split menjadi list angka
    rows = cleaned_rows.split(",")

    print("Extracted OCR after cleaning:", rows)  # Debugging

    # Pastikan jumlah baris tepat 15
    if len(rows) > 15:
        rows = rows[:15]  # Ambil hanya 15 baris pertama
    elif len(rows) < 15:
        rows.extend([""] * (15 - len(rows)))  # Tambahkan baris kosong

    valid_rows = []
    all_validity = []
    is_valid = True  # Default valid

    for row in rows:
        row = row.strip()  # Hilangkan spasi di awal/akhir

        # Hapus semua karakter selain digit
        row = re.sub(r"[^\d]", "", row)

        # Potong kalau lebih dari 3
        if len(row) > 3:
            row = row[:3]

        # Padding di sini supaya tiap row pasti 3 karakter
        row = row.ljust(3, ' ')

        # Cek validitas per karakter
        for char in row:
            if char.isdigit():
                all_validity.append(True)
            else:
                all_validity.append(False)
                is_valid = False

        valid_rows.append(row)

    # Gabungkan semuanya
    cleaned_output = list("".join(valid_rows))

    for i in range(len(cleaned_output)):
        if cleaned_output[i] != " ":
            cleaned_output[i] = int(cleaned_output[i])

    return is_valid, cleaned_output, all_validity

def validate_loss_2(ocr_output):
    print("Extracted OCR before cleaning:", ocr_output)  # Debugging

    # Hapus karakter selain digit, spasi, dan koma
    cleaned_rows = re.sub(r"[^\d,]", "", ocr_output)

    # Pastikan koma tidak berlebihan (hapus koma di awal/akhir)
    cleaned_rows = cleaned_rows.strip(",")

    # Split menjadi list angka
    rows = cleaned_rows.split(",")

    print("Extracted OCR after cleaning:", rows)  # Debugging

    # Pastikan jumlah baris tepat 13
    if len(rows) > 13:
        rows = rows[:13]  # Ambil hanya 13 baris pertama
    elif len(rows) < 13:
        rows.extend([""] * (13 - len(rows)))  # Tambahkan baris kosong

    valid_rows = []
    all_validity = []
    is_valid = True  # Default valid

    for row in rows:
        row = row.strip()  # Hilangkan spasi di awal/akhir

        # Hapus semua karakter selain digit
        row = re.sub(r"[^\d]", "", row)

        # Potong kalau lebih dari 2
        if len(row) > 2:
            row = row[:2]

        # Padding di sini supaya tiap row pasti 3 karakter
        row = row.ljust(2, ' ')

        # Cek validitas per karakter
        for char in row:
            if char.isdigit():
                all_validity.append(True)
            else:
                all_validity.append(False)
                is_valid = False

        valid_rows.append(row)

    # Gabungkan semuanya
    cleaned_output = list("".join(valid_rows))

    for i in range(len(cleaned_output)):
        if cleaned_output[i] != " ":
            cleaned_output[i] = int(cleaned_output[i])

    return is_valid, cleaned_output, all_validity

def validate_time(ocr_output):
    # Hapus karakter selain angka
    cleaned_output = re.sub(r"\D", "", ocr_output)

    # Jika kosong setelah dibersihkan, ganti dengan "0000"
    if not cleaned_output:
        cleaned_output = "0000"
    
    # Pastikan panjang hasil sesuai format (4 digit)
    if len(cleaned_output) > 4:
        cleaned_output = cleaned_output[:4]  # Ambil 4 karakter pertama
    elif len(cleaned_output) < 4:
        cleaned_output = cleaned_output.ljust(4, '0')  # Padding dengan '0' di kanan

    # Ambil jam (HH) dan menit (MM)
    hour = int(cleaned_output[:2])
    minute = int(cleaned_output[2:])

    # Cek apakah jam dan menit dalam rentang valid
    is_valid = (0 <= hour <= 23) and (0 <= minute <= 59)

    # Konversi string ke list integer
    cleaned_list = [int(digit) for digit in cleaned_output]

    # Tandai validitas karakter setelah diproses
    character_validity = [is_valid] * len(cleaned_output)

    return is_valid, [int(d) for d in cleaned_list], character_validity

def validate_rest(ocr_output):
    # Bersihkan output dengan menghapus spasi & newline, lalu ubah ke uppercase
    cleaned_output = ocr_output.strip().upper()

    # Pastikan panjang hasil sesuai format (1 digit)
    if len(cleaned_output) > 1:
        cleaned_output = cleaned_output[:1]
    elif len(cleaned_output) < 1:
        cleaned_output = cleaned_output.ljust(1, ' ')

    # Pastikan output HANYA 1 karakter dan harus 'O' atau 'X'
    if cleaned_output in ["O", "X"] and len(cleaned_output) == 1:
        return True, list(cleaned_output), [True]
    else:
        return False, list(cleaned_output), [False]

def validate_single_num(ocr_output):
    # Hapus karakter selain angka
    cleaned_output = re.sub(r"\D", "", ocr_output)  # Hanya simpan angka

    # Jika kosong setelah dibersihkan, langsung return False
    if not cleaned_output:
        return False, [0], [False]  # Mengembalikan angka 0 sebagai default

    # Ambil hanya 1 digit pertama jika lebih dari 1 angka
    cleaned_output = int(cleaned_output[0])  # Konversi ke integer
    output_final = []
    output_final.append(cleaned_output)

    return True, output_final, [True]

def validate_single_loss1(ocr_output):
    print("Extracted OCR before cleaning:", ocr_output)  # Debugging

    # Hapus karakter selain digit dan spasi
    cleaned_output = re.sub(r"[^\d\s]", "", ocr_output)

    all_validity = []
    is_valid = True  # Default valid

    cleaned_output = cleaned_output.strip()  # Hilangkan spasi di awal/akhir

    # Pastikan panjang sesuai format (tepat 3 karakter setelah pembersihan)
    if len(cleaned_output) > 3:
        cleaned_output = cleaned_output[:3]
    else:
        cleaned_output = cleaned_output.ljust(3, ' ')

    # Cek apakah formatnya sesuai: 'N  ', 'NN ', 'NNN', atau '   '
    if not re.fullmatch(r"\d\s{2}|\d{2}\s|\d{3}|\s{3}", cleaned_output):
        print(f"Kondisi IF terpenuhi untuk: '{cleaned_output}'")
        all_validity.extend([False] * 3)
        is_valid = False
    else:
        print(f"Kondisi ELSE terpenuhi untuk: '{cleaned_output}'")
        all_validity.extend([True] * 3)

    cleaned_output_list = list(cleaned_output)

    for i in range(0, len(cleaned_output_list)):
        if cleaned_output_list[i] != " ":
            try:
                cleaned_output_list[i] = int(cleaned_output_list[i])
            except ValueError:
                print(f"Error converting '{cleaned_output_list[i]}' to integer.")
                is_valid = False
                if len(all_validity) > i:
                    all_validity[i] = False
                elif len(all_validity) < 3:
                    all_validity.append(False) # Tambah jika belum cukup panjang

    return is_valid, cleaned_output_list, all_validity

def validate_single_loss2(ocr_output):
    print("Extracted OCR before cleaning:", ocr_output)  # Debugging

    # Hapus karakter selain digit dan spasi
    cleaned_output = re.sub(r"[^\d\s]", "", ocr_output)

    all_validity = []
    is_valid = True  # Default valid

    cleaned_output = cleaned_output.strip()  # Hilangkan spasi di awal/akhir

    # Pastikan panjang sesuai format (maksimal 2 karakter setelah pembersihan)
    if len(cleaned_output) > 2:
        cleaned_output = cleaned_output[:2]
    else:
        cleaned_output = cleaned_output.ljust(2, ' ')

    # Cek apakah formatnya sesuai: 'N ', 'NN', atau '  '
    if not re.fullmatch(r"\d\s|\d{2}|\s{2}", cleaned_output):
        print(f"Kondisi IF terpenuhi untuk: '{cleaned_output}'")
        all_validity.extend([False] * 2)
        is_valid = False
    else:
        print(f"Kondisi ELSE terpenuhi untuk: '{cleaned_output}'")
        all_validity.extend([True] * 2)

    cleaned_output_list = list(cleaned_output)

    for i in range(0, len(cleaned_output_list)):
        if cleaned_output_list[i] != " ":
            try:
                cleaned_output_list[i] = int(cleaned_output_list[i])
            except ValueError:
                print(f"Error converting '{cleaned_output_list[i]}' to integer.")
                is_valid = False
                if len(all_validity) > i:
                    all_validity[i] = False
                elif len(all_validity) < 2:
                    all_validity.append(False) # Tambah jika belum cukup panjang

    return is_valid, cleaned_output_list, all_validity

def validate_single_lot(ocr_output):
    # Konversi input ke huruf besar
    ocr_output = ocr_output.upper()
    ocr_output = re.sub(r'[^a-zA-Z0-9]', '', ocr_output)

    all_validity = []
    is_valid = True

    # Hapus semua spasi
    ocr_output = ocr_output.replace(' ', '')

    # Pastikan panjang 6 karakter
    if len(ocr_output) > 6:
        ocr_output = ocr_output[:6]
    else:
        ocr_output = ocr_output.ljust(6, ' ')

    ocr_output_list = list(ocr_output)

    # Cek per karakter sesuai posisinya
    for j, char in enumerate(ocr_output_list):
        if j < 3 and char.isalpha():  # Posisi 0,1,2 harus huruf
            all_validity.append(True)
        elif j == 3 and char.isdigit():  # Posisi 3 harus angka
            all_validity.append(True)
        elif j == 4 and char.isalpha():  # Posisi 4 harus huruf
            all_validity.append(True)
        elif j == 5 and char.isdigit():  # Posisi 5 harus angka
            all_validity.append(True)
        elif char == " ":
            all_validity.append(True)
        elif (j < 3 and not char.isalpha()) or (j < 3 and char != " "):
            if char == 1 or char == "1":
                char = "I"
            elif char == 2 or char == "2":
                char = "Z"
            elif char == 4 or char == "4":
                char = "A"
            elif char == 5 or char == "5":
                char == "S"
            elif char == 6 or char == "6":
                char == "G"
            elif j == 7 or char == "7":
                char == "T"
            elif char == 8 or char == "8":
                char == "B" 
            ocr_output_list[j] = char
            all_validity.append(False)
            is_valid = False
        else:
            all_validity.append(False)
            is_valid = False

    # Konversi angka di posisi 3 dan 5 ke integer
    for j in [3, 5]:
        if j < len(ocr_output_list) and ocr_output_list[j].isdigit():
            ocr_output_list[j] = int(ocr_output_list[j])
        else:
            ocr_output_list[j] = "#"

    return is_valid, ocr_output_list, all_validity

# Simpan model sebagai fungsi yang dapat dipanggil
def recognize_text(image, category, ocr_prompts):
    print("successfully get into function")
    match category:
        case "main_filler":
            print("processing main filler...")
            ocr_final = process_image_with_gemini(image, ocr_prompts['main_filler'])
            is_valid, result, validity_array = validate_main_filler(ocr_final)

        case "implementer":
            print("processing implementer...")
            ocr_final = process_image_with_gemini(image, ocr_prompts['implementer'])
            is_valid, result, validity_array = validate_implementer(ocr_final)

        case "date":
            print("processing date...")
            ocr_final = process_image_with_gemini(image, ocr_prompts['date'])
            print(ocr_final)
            is_valid, result, validity_array = validate_date(ocr_final)

        case "product_tank":
            print("processing tank...")
            ocr_final = process_image_with_gemini(image, ocr_prompts['product_tank'])
            is_valid, result, validity_array = validate_product_tank(ocr_final)

        case "volume":
            print("processing volume...")
            ocr_final = process_image_with_gemini(image, ocr_prompts['volume'])
            is_valid, result, validity_array = validate_volume(ocr_final)

        case "volume_ccp":
            print("processing ccp...")
            ocr_final = process_image_with_gemini(image, ocr_prompts['volume_ccp'])
            is_valid, result, validity_array = validate_volume_ccp(ocr_final)

        # case "lot1":
        #     print("processing lot1...")
        #     ocr_final = process_image_with_gemini(image, ocr_prompts['lot1'])
        #     is_valid, result, validity_array = validate_lot1(ocr_final)

        # case "lot2":
        #     print("processing lot2...")
        #     ocr_final = process_image_with_gemini(image, ocr_prompts['lot2'])
        #     is_valid, result, validity_array = validate_lot2(ocr_final)

        # case "loss1":
        #     print("processing loss1...")
        #     ocr_final = process_image_with_gemini(image, ocr_prompts['loss1'])
        #     is_valid, result, validity_array = validate_loss_1(ocr_final)

        # case "loss2":
        #     print("processing loss2...")
        #     ocr_final = process_image_with_gemini(image, ocr_prompts['loss2'])
        #     is_valid, result, validity_array = validate_loss_2(ocr_final)

        case "time":
            print("processing time...")
            ocr_final = process_image_with_gemini(image, ocr_prompts['time'])
            is_valid, result, validity_array = validate_time(ocr_final)

        case "rest":
            print("processing rest...")
            ocr_final = process_image_with_gemini(image, ocr_prompts['rest'])
            is_valid, result, validity_array = validate_rest(ocr_final)

        case "single_num":
            print("processing single num...")
            ocr_final = process_image_with_gemini(image, ocr_prompts['single_num'])
            is_valid, result, validity_array = validate_single_num(ocr_final)
        
        case "loss1":
            print("processing single loss1...")
            ocr_final = process_image_with_gemini(image, ocr_prompts['single_loss1'])
            is_valid, result, validity_array = validate_single_loss1(ocr_final)
        
        case "loss2":
            print("processing single loss2...")
            ocr_final = process_image_with_gemini(image, ocr_prompts['single_loss2'])
            is_valid, result, validity_array = validate_single_loss2(ocr_final)

        case "lot1":
            print("processing single lot...")
            ocr_final = process_image_with_gemini(image, ocr_prompts['single_lot'])
            print(ocr_final)
            is_valid, result, validity_array = validate_single_lot(ocr_final)

        case _:
            raise ValueError("Invalid category.")

    return result, is_valid, validity_array