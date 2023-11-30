# import pandas as pd
# import matplotlib.pyplot as plt

# # Đọc dữ liệu từ file CSV
# data = pd.read_csv('output_0.csv')

# x_data = data['x']
# y_data = data['y']
# window_side = 20

# mean_values = [0 for _ in range(window_side)]  # Danh sách lưu trữ giá trị trung bình từ 0 đến i

# for i in range(window_side, len(y_data)):
#     mean_values.append(y_data[i-window_side:i+1].mean())  # Tính trung bình từ 0 đến chỉ số i

# plt.figure(figsize=(8, 6))
# plt.plot(x_data, y_data, marker='o', linestyle='-', label='y values')
# plt.plot(x_data, mean_values, color='r', linestyle='--', label='Mean from 0 to i')  # Vẽ đường trung bình từ 0 đến i
# plt.title('Biểu đồ x và y từ file CSV')
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')
# plt.legend()
# plt.grid(True)
# plt.show()

import os
import pandas as pd
import matplotlib.pyplot as plt

# Lấy danh sách tất cả các file trong thư mục hiện tại
files = [file for file in os.listdir() if file.endswith('.csv')]

# Tạo danh sách lưu trữ tên file và các giá trị trung bình từ mỗi file
file_mean_values = []

window_side = 20

# Đọc và tính toán giá trị trung bình từng file
for file in files:
    data = pd.read_csv(file)
    y_data = data['y']

    mean_values = [0 for _ in range(window_side)]  # Danh sách lưu trữ giá trị trung bình từ 0 đến i

    for i in range(window_side, len(y_data)):
        mean_values.append(y_data[i-window_side:i+1].mean())  # Tính trung bình từ 0 đến chỉ số i

    file_mean_values.append((file, mean_values))

# Vẽ các đường trung bình từ mỗi file lên cùng một biểu đồ
plt.figure(figsize=(8, 6))
for file, mean_values in file_mean_values:
    plt.plot(mean_values, linestyle='--', label=f'{file[:-4]} Mean values')  # Lấy tên file (loại bỏ phần mở rộng .csv)

plt.title('Các đường trung bình từ các file CSV')
plt.xlabel('Index')
plt.ylabel('Mean Y Values')
plt.legend()
plt.grid(True)
plt.show()
