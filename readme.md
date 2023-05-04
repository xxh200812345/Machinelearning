# 护照windows部署手册

## 部署OCR程序
从github链接clone项目到windows 
https://github.com/xxh200812345/Machinelearning.git

获得如下文件目录

    Machinelearning/
    Mode                 LastWriteTime         Length Name
    ----                 -------------         ------ ----
    d-----         2023/4/28     19:29                jpocr
    -a----         2023/4/28     19:29           8196 .DS_Store
    -a----         2023/4/28     19:29             41 .gitignore
    -a----         2023/4/28     19:29           9247 MNIST_main.py
    -a----         2023/4/28     19:30            641 readme.md
    -a----         2023/4/28     19:29           2520 tesseract.py


## 安装tessract
https://www.jianshu.com/p/f7cb0b3f337a
根据教程下载安装windows版本

命令行使用 `tesseract --list-langs` 命令可查看当前软件支持的语言

    List of available languages in "E:\ProgramFiles\Tesseract-OCR\jTessBoxEditor\tesseract-ocr\tessdata/" (2):
    eng
    jpn
    jpn_vert
    osd

正确打印便证明安装成功

### 配置虚拟环境

    cd .\jpocr\
    python -m venv ai_jpocr_venv
    ai_jpocr_venv\Scripts\activate
    pip install -r requirements.txt

正确安装输出log：

    Successfully installed Pillow-9.5.0 autopep8-2.0.2 black-23.3.0 click-8.1.3 colorama-0.4.6 contourpy-1.0.7 cycler-0.11.0 fonttools-4.39.3 importlib-resources-5.12.0 kiwisolver-1.4.4 matplotlib-3.7.1 mypy-extensions-1.0.0 numpy-1.24.2 opencv-contrib-python-4.7.0.72 opencv-python-4.7.0.72 packaging-23.0 pathspec-0.11.1 platformdirs-3.2.0 pycodestyle-2.10.0 pyocr-0.7.2 pyparsing-3.0.9 python-dateutil-2.8.2 six-1.16.0 tomli-2.0.1 typing_extensions-4.5.0 zipp-3.15.0

### 设置安装tessract的应用路径

打开 `jpocr/cor_configs.ini` 文件

如果是windows系统

> 修改 WINDOWS_TESSRACT_LOCATION

如果是mac系统

> 修改 MAC_TESSRACT_LOCATION

输入
1. 复制 `jpocr/passport_imgs`绝对路径
1. 修改 PASSPORT_IMAGES_FOLDER_PATH

输出
1. 复制 `jpocr/output`绝对路径
1. 修改 OUTPUT_FOLDER_PATH

### 设置白名单

1. 找到本机安装的 `Tesseract-OCR` 的文件夹
1. 打开 `Tesseract-OCR\tessdata\configs\` 文件夹
1. 复制 `res/_my_word` 文件到 `configs` 文件夹

### 设置自定义模型

1. 找到本机安装的 `Tesseract-OCR` 的文件夹
1. 打开 `Tesseract-OCR\tessdata\` 文件夹
1. 复制模型 `jpocr\res\num_1.traineddata` 到 `tessdata\`文件夹
1. 使用 `tesseract --list-langs` 命令,查看是否正确安装：eng jpn jpn_vert `num_1` osd

### 运行

    (ai_jpocr_venv) PS H:\vswork\Machinelearning\jpocr> python main.py

## 测试
### 输入图片标准说明
参考 `jpocr\passport_imgs\sample.jpg`

### 输出结果以及说明
jpocr\output\

| 文件名 | 说明 |
| --- | --- |
| *_edited.png | 图像处理后的图像 |
| *_sign.png | 签名图像 |
| *_tessract.png | OCR图像 |
| data.json | 存放处理后的护照信息 |
| text_imgs | 存放切割后的文字块的文件夹 |

## 训练

1. 打包图片成字库 `num_1.font.exp0.tif`
1. 生成box。`tesseract num_1.font.exp0.tif num_1.font.exp0 –l eng batch.nochop makebox`

训练命令

    echo "font 0 0 0 0 0">font_properties
    tesseract num_1.font.exp0.tif num_1.font.exp0 nobatch box.train
    unicharset_extractor num_1.font.exp0.box
    mftraining -F font_properties -U unicharset -O num_1.unicharset num_1.font.exp0.tr
    cntraining num_1.font.exp0.tr

    mv inttemp num_1.inttemp
    mv pffmtable num_1.pffmtable
    mv normproto num_1.normproto
    mv shapetable num_1.shapetable

    combine_tessdata num_1.

### Tesseract-OCR 5.0LSTM训练流程
https://www.cnblogs.com/nayitian/p/15240143.html

tesseract num_1.font.exp0.tif num_1.font.exp0.box --psm 6 lstm.train
combine_tessdata -e eng.traineddata eng.lstm
lstmtraining --model_output="output" --continue_from="eng.lstm" --train_listfile="num.training_files" --traineddata="eng.traineddata" --debug_interval -1 --max_iterations 800
lstmtraining --stop_training --continue_from="output_checkpoint" --traineddata="eng.traineddata" --model_output="num_1.traineddata"