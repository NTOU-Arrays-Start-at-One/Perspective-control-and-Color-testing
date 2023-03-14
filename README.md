# Perspective-control-and-Color-testing

此專案的目的在於比較色彩還原後的影像與其原始影像的色彩差異，並透過分析兩張影像中色板來達成此目的。

This project aims to compare the color difference between color-restored images and their original counterparts. The actual process involves analyzing the color palette differences between the two images.

![unwarp_restored_model and Original_photo](https://github.com/NTOU-Arrays-Start-at-One/Perspective-control-and-Color-testing/blob/main/result/result/unwarp_restored_model%20and%20Original_photo.png?raw=true)
![Histogram of delta_e_1 and delta_e_2.png](https://github.com/NTOU-Arrays-Start-at-One/Perspective-control-and-Color-testing/blob/main/result/result/Histogram%20of%20delta_e_1%20and%20delta_e_2.png?raw=true)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15IAx8eKlwPET7O_HTYrmF4C4BVypd9dP?usp=sharing)
[ipynb](色板校正與分析_2.ipynb)
## Installation

+ 若要在您的本地端執行此專案，請依照以下步驟進行：

1. 執行以下指令來複製此專案的 repository：
```bash
git clone https://github.com/NTOU-Arrays-Start-at-One/Perspective-control-and-Color-testing.git
```

2. 執行以下指令安裝所需的套件：
```bash
pip install -r requirements.txt
```
這個指令會安裝此專案所需的所有套件。

<hr>

+ To run this project on your local machine, follow these steps:

1. Clone the repository by running the following command:
```bash
git clone https://github.com/NTOU-Arrays-Start-at-One/Perspective-control-and-Color-testing.git
```

2. Install the required dependencies by running the following command:
```bash
pip install -r requirements.txt
```
This will install all the necessary packages for this project.

## Usage

+ 執行此專案，請按照以下步驟執行程式：
1. 將原始的圖板影像放入 `src/Original` 。
2. 將還原後的圖板影像放入 `src/Result_restored` 。
3. 執行 `python try.py` 以顯示色差的比較與結果。
4. 執行完成後，文字與圖片會儲存於 `result` 中。

請按照以上步驟執行程式。如有任何問題，請聯絡開發人員。

<hr>

+ To use this project, follow these steps:
1. Place the original canvas image in the `src/Original directory` .
2. Place the restored canvas image in the `src/Result_restored directory` .
3. Run `python try.py` to display a comparison of the color difference and results.
4. Upon completion, the text and images will be stored in the `result` directory.

Please follow the above steps to run the program. If you encounter any issues, contact the developers.

![](https://github.com/NTOU-Arrays-Start-at-One/Perspective-control-and-Color-testing/blob/main/result/result/delta_e_1_unwarp_restored_model.png?raw=true)
![](https://github.com/NTOU-Arrays-Start-at-One/Perspective-control-and-Color-testing/blob/main/result/result/delta_e_1.png?raw=true)

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository and clone it to your local machine.
2. Create a new branch and make your changes.
3. Push your changes to your forked repository.
4. Create a pull request to merge your changes into the main repository.
