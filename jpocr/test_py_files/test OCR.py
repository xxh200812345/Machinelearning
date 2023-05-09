import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyocr
import pyocr.builders
import cv2
import platform

import os
from PIL import Image

def get_file_paths(folder_path):
    """
    获取文件夹中所有文件路径
    :param folder_path: 文件夹路径
    :return: 文件路径列表
    """
    file_paths = []  # 文件路径列表
    # 获取文件夹中所有文件名
    file_names = os.listdir(folder_path)
    # 遍历所有文件名
    for file_name in file_names:

        if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
            # 拼接文件路径
            file_path = os.path.join(folder_path, file_name)
            # 判断文件是否为文件夹
            if os.path.isdir(file_path):
                # 如果是文件夹，则递归调用该函数
                sub_file_paths = get_file_paths(file_path)
                file_paths.extend(sub_file_paths)
            else:
                # 如果是文件，则将文件路径添加到文件路径列表中
                file_paths.append(file_path)
    return file_paths


# 目标图
sample_img_path = "output/sample.png"
sample_edited_img_path = "output/sample_edited.png"
sample_sign_img_path = "output/sample_sign.png"

# 获取操作系统名称及版本号
pyocr.tesseract.TESSERACT_CMD = "E:/Program Files/Tesseract-OCR/tesseract.exe"

# 判断当前操作系统
if platform.system() == "Darwin":
    pyocr.tesseract.TESSERACT_CMD = "/opt/homebrew/Cellar/tesseract/5.3.1/bin/tesseract"

def ocr_by_key(img, key):
    # ツール取得
    tool = pyocr.get_available_tools()[0]

    # OCR
    builder = None
    digits = None

    if key == "line":
        builder = pyocr.builders.LineBoxBuilder(tesseract_layout=6)
        builder.tesseract_configs.append("_my_word_jp")

    elif key == "word":
        builder = pyocr.builders.WordBoxBuilder(tesseract_layout=6)
        builder.tesseract_configs.append("_my_word_jp")

    else:
        print("key is error.")

    digits = tool.image_to_string(Image.fromarray(img), lang="jpn", builder=builder)

    return digits



# 识别后数据记录
def rect_set(img, data_list):
    for data in data_list:
        rect = data.position
        text = data.content
        print(f"text：{text}   rect：{rect}")
    # 遍历每个矩形，绘制在图片上
    for data in data_list:
        rect = data.position
        x1, y1, x2, y2 = rect[0][0], rect[0][1], rect[1][0], rect[1][1]
        cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 255), 2)

    # 将图像从OpenCV格式转换为Pillow格式
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # 遍历每个矩形，绘制在图片上
    for data in data_list:
        rect = data.position
        x1, y1, x2, y2 = rect[0][0], rect[0][1], rect[1][0], rect[1][1]
        # 定义要绘制的文本和位置
        text = data.content
        # 定义文本样式
        font_path = "res/NotoSansJP-Thin.otf"
        font_size = 16
        font_color = (0, 0, 255)

        # 加载日语字体
        font = ImageFont.truetype(font_path, font_size)

        # 在图像上绘制文本
        draw.text((x1, y1 - 23), text, font=font, fill=font_color)

    # 将图像从Pillow格式转换为OpenCV格式
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # cv2.imshow("img with rectangles", img_cv)
    # cv2.waitKey(0)

    return img_cv

# 固定色阶
def color_scale_display(img, Shadow=0, Highlight=255, Midtones=1):
    """
    用于图像预处理，模拟ps的色阶调整
    img：传入的图片
    Shadow：黑场(0-Highlight)
    Highlight：白场(Shadow-255)
    Midtones：灰场(9.99-0.01)
    0 <= Shadow < Highlight <= 255
    返回一张图片
    """
    if Highlight > 255:
        Highlight = 255
    if Shadow < 0:
        Shadow = 0
    if Shadow >= Highlight:
        Shadow = Highlight - 2
    if Midtones > 9.99:
        Midtones = 9.99
    if Midtones < 0.01:
        Midtones = 0.01
    # 转类型
    img = np.array(img, dtype=np.float16)
    # 计算白场黑场离差
    Diff = Highlight - Shadow
    img = img - Shadow
    img[img < 0] = 0
    img = (img / Diff) ** (1 / Midtones) * 255
    img[img > 255] = 255
    # 转类型
    img = np.array(img, dtype=np.uint8)
    return img

def resize_image_half(image):

    # 获取图像宽度和高度
    width = int(image.shape[1] * 0.5)
    height = int(image.shape[0] * 0.5)

    # 将图像缩小到一半
    resized_image = cv2.resize(image, (width, height))

    # 返回调整后的图像
    return resized_image

if __name__ == "__main__":

    pdf2png='/Users/xianhangxiao/vswork/Machinelearning/jpocr/output'
    paths=get_file_paths(pdf2png)
    for path in paths:

        file_name=os.path.basename(path)
        name_without_extension = os.path.splitext(file_name)[0]

        # 裁切不需要的部分,返回处理后的
        img = cv2.imread(path)
        img= resize_image_half(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = color_scale_display(gray, 112, 217, 0.97)

        cv2.imwrite(f"{pdf2png}/{name_without_extension}_thresh.png", thresh)

        data_list = ocr_by_key(thresh,"word")

        # cv2.imshow("img with rectangles", thresh)
        # cv2.waitKey(0)

        # 显示绘制好矩形的图片
        img_cv=rect_set(img, data_list)

        cv2.imwrite(f"{pdf2png}/{name_without_extension}.png", img_cv)


