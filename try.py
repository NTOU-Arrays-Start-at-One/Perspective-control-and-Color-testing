import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # 表格顏色
import cba  # cba: ColorBlock Analysis

#-------------------------------------------------------------------------#
# 色板影像校正
#-------------------------------------------------------------------------#
standard_val, standard_unwarp = cba.correction_and_analysis(cv2.imread("src/Standard.png"))
original_photo_val, original_photo_unwarp = cba.correction_and_analysis(cv2.imread("src/Original.jpg"))
restored_val, restored_unwarp = cba.correction_and_analysis(cv2.imread("src/Result_restored.jpg"))

#-------------------------------------------------------------------------#
# 比較兩色版的差異
#-------------------------------------------------------------------------#
# 顯示校正後的色板
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(cv2.cvtColor(standard_unwarp, cv2.COLOR_BGR2RGB))
ax1.set_title('Standard')
ax2.imshow(cv2.cvtColor(restored_unwarp, cv2.COLOR_BGR2RGB))
ax2.set_title('Restored')
plt.show()

# 測試與比較
delta_e_1 = cba.compare_colorboard(standard_val, restored_val)

#-----------
# 顯示校正後的色板
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(cv2.cvtColor(standard_unwarp, cv2.COLOR_BGR2RGB))
ax1.set_title('Standard')
ax2.imshow(cv2.cvtColor(original_photo_unwarp, cv2.COLOR_BGR2RGB))
ax2.set_title('Original photo')
plt.show()

# 測試與比較
delta_e_2 = cba.compare_colorboard(standard_val, original_photo_val)

#-----------
# 顯示
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
ax1.imshow(cv2.cvtColor(standard_unwarp, cv2.COLOR_BGR2RGB))
ax1.set_title('Standard')
ax2.imshow(cv2.cvtColor(original_photo_unwarp, cv2.COLOR_BGR2RGB))
ax2.set_title('Original_photo')
ax3.imshow(cv2.cvtColor(restored_unwarp, cv2.COLOR_BGR2RGB))
ax3.set_title('restored_model')

# 計算平均值
delta_e_1 = np.array(delta_e_1)
delta_e_2 = np.array(delta_e_2)
delta_e_1_mean = np.mean(delta_e_1) # restored model
delta_e_2_mean = np.mean(delta_e_2) # Original_photo

# 繪製直方圖
x = np.arange(0, 25)
labels = [f"({i//5},{i%5})" for i in range(25)]
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x, delta_e_1.reshape(25), width=0.4, label='delta_e_1: restored_model')
ax.bar(x + 0.4, delta_e_2.reshape(25), width=0.4, label='delta_e_2: Original_photo')

# 繪製平均值
ax.axhline(delta_e_1_mean, color='r', linestyle='--', label='delta_e_1 mean')
ax.axhline(delta_e_2_mean, color='g', linestyle='--', label='delta_e_2 mean')

ax.text(24.6, delta_e_1_mean, f"{delta_e_1_mean:.2f}", ha='right', va='bottom')
ax.text(24.6, delta_e_2_mean, f"{delta_e_2_mean:.2f}", ha='right', va='top')

# 設定圖表屬性
ax.set_xticks(x+0.4/2)
ax.set_xticklabels(labels)
ax.legend()
ax.set_title('Histogram of delta_e_1 and delta_e_2')
ax.set_xlabel('(i,j)')
ax.set_ylabel('Value')

# print(np.mean(delta_e_1))
# print(np.mean(delta_e_2))

plt.show()