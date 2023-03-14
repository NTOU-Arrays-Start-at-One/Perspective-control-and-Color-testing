# Perspective-control-and-Color-testing

此專案的目的在於比較色彩還原後的影像與其原始影像的色彩差異，並透過分析兩張影像中色板來達成此目的。

This project aims to compare the color difference between color-restored images and their original counterparts. The actual process involves analyzing the color palette differences between the two images.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15IAx8eKlwPET7O_HTYrmF4C4BVypd9dP?usp=sharing)
[ipynb](色板校正與分析_2.ipynb)
## Installation

To run this project on your local machine, follow these steps:

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

執行此專案，請按照以下步驟執行程式

To use this project, follow these steps:

1. 將原始的圖板影像放入 `src/Original` 。
2. 將還原後的圖板影像放入 `src/Result_restored` 。
3. 執行 `python try.py` 以顯示色差的比較與結果。
4. 執行完成後，文字與圖片會儲存於 `result` 中。

1. Place the original canvas image in the `src/Original directory` .
2. Place the restored canvas image in the `src/Result_restored directory` .
3. Run `python try.py` to display a comparison of the color difference and results.
4. Upon completion, the text and images will be stored in the `result` directory.

請按照以上步驟執行程式。如有任何問題，請聯絡開發人員。

Please follow the above steps to run the program. If you encounter any issues, contact the developers.

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository and clone it to your local machine.
2. Create a new branch and make your changes.
3. Push your changes to your forked repository.
4. Create a pull request to merge your changes into the main repository.
