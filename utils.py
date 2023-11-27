import numpy as np
import random


class ActionSpace(object):
    def __init__(self, number_action):
        self.n = number_action
    def sample(self):
        return random.randint(0, self.n - 1)

def convert_base(number, base, length = 1):
        if number == 0:
            return [0]*length
        
        digits = []
        negative = False
        
        if number < 0:
            negative = True
            number = abs(number)
        
        while number > 0:
            remainder = number % base
            digits.append(int(remainder))
            number //= base

        if length > len(digits):
            for _ in range(length - len(digits)):
                digits.append(0)
        
        if negative:
            digits.append("-")

        digits = digits[::-1]

        return digits

def print_array( string="", arr=[]):
    return
    print(f"{string:12s}", end=" [")
    for item in arr:
        str_ = f"{item:8.2f}"
        print(str_, end=" ")
    print("]")

def get_latest_file(directory_path="./"):
    import os

    # List all files in the directory
    all_files = os.listdir(directory_path)

    # Filter only files (not directories)
    file_paths = [os.path.join(directory_path, file) for file in all_files if os.path.isfile(os.path.join(directory_path, file))]

    # Sort the files by creation time (descending order)
    file_paths.sort(key=lambda x: os.path.getctime(x), reverse=True)

    # Get the latest created file
    latest_file = file_paths[0] if file_paths else None
    if latest_file:
        # Get the name of the latest created file
        latest_file = os.path.basename(latest_file)
    return latest_file

def get_all_file(directory_path="./"):
    import os

    # List all files in the directory
    all_files = os.listdir(directory_path)

    # Filter only files (not directories)
    file_paths = [os.path.join(directory_path, file) for file in all_files if os.path.isfile(os.path.join(directory_path, file))]

    # Sort the files by creation time (descending order)
    # file_paths.sort(key=lambda x: os.path.getctime(x), reverse=True)

    list_f = []
    for f_name in file_paths:
        # Get the name of the latest created file
        f_base = os.path.basename(f_name)
        list_f.append(f_base)
    return list_f


def get_new_file_name(file_name="data.txt",folder_path="./"):
    import os
    file_path = os.path.join(folder_path, file_name)
    new_file_name = file_name
    if os.path.exists(file_path):
        # Nếu tệp đã tồn tại, tạo tên mới với số 0, 1, 2,...
        i = 0
        while True:
            new_file_name = f"{os.path.splitext(file_name)[0]}_{i}{os.path.splitext(file_name)[1]}"
            new_file_path = os.path.join(folder_path, new_file_name)
            if not os.path.exists(new_file_path):
                break
            i += 1
    
    return new_file_name
    

def parse_command_line_args():
    import argparse
    # Tạo một đối tượng ArgumentParser
    parser = argparse.ArgumentParser(description="My Python Script")

    # Thêm đối số dòng lệnh cho tên môi trường (-name) với giá trị mặc định
    parser.add_argument('-name', '--env_name', type=str, default="IoT", help="The environment name")
    parser.add_argument('-type', '--type_show', type=str, default="one", help="The option show: 1 line, 2  line, all line")

    # Phân tích các đối số
    args = parser.parse_args()

    # Trả về giá trị của đối số -name hoặc --env_name
    return args

def get_all_file(directory_path="./"):
    import os

    # List all files in the directory
    all_files = os.listdir(directory_path)

    # Filter only files (not directories)
    file_paths = [os.path.join(directory_path, file) for file in all_files if os.path.isfile(os.path.join(directory_path, file))]

    # Sort the files by creation time (descending order)
    # file_paths.sort(key=lambda x: os.path.getctime(x), reverse=True)

    list_f = []
    for f_name in file_paths:
        # Get the name of the latest created file
        f_base = os.path.basename(f_name)
        list_f.append(f_base)
    return list_f