import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyocr
import pyocr.builders
import cv2
import platform
from collections import Counter

import os
from PIL import Image

from passport import Passport


# 返回非空白区域范围
def calc_row_and_col_sum(img):
    # 计算每行的值
    row_sum = np.sum(img, axis=1)

    # 找到出现次数最多的值
    max_val = np.max(row_sum)

    # 行值数组减去出现次数最多的值
    row_sum = max_val - row_sum

    # 创建一个布尔数组，指示哪些元素大于 99999
    mask = row_sum > 9999

    # 使用布尔索引和 np.where() 函数找到满足条件的行号
    row_numbers = np.where(mask)[0]

    # 计算每列的值
    col_sum = np.sum(img, axis=0)

    # 找到出现次数最多的值
    max_val = np.max(col_sum)

    # 列值数组减去出现次数最多的值
    col_sum = max_val - col_sum

    # 创建一个布尔数组，指示哪些元素大于 8888
    mask = col_sum > 9999

    # 使用布尔索引和 np.where() 函数找到满足条件的列号
    col_numbers = np.where(mask)[0]

    # 返回矩形范围
    return (
        col_numbers.min(),
        row_numbers.min(),
        col_numbers.max(),
        row_numbers.max() + 20,
    )


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


# 去除图片周围边框
def crop_image(img, key):
    # 获取图像的尺寸
    height, width, channels = img.shape

    # 计算截取的左、右、上、下边界
    left = key
    right = width - key
    top = key
    bottom = height - key

    # 使用切片操作截取图像的中心部分
    center = img[top:bottom, left:right]

    # 返回裁剪后的图像
    return center


# 设置tessract入口程序安装位置
def set_tessract_app():
    # 获取操作系统名称及版本号
    os = platform.system()
    pyocr.tesseract.TESSERACT_CMD = config_options["WINDOWS_TESSRACT_LOCATION"]

    # 判断当前操作系统
    if os == "Darwin":
        pyocr.tesseract.TESSERACT_CMD = config_options["MAC_TESSRACT_LOCATION"]


# 初始化设置
def init(passport: Passport):
    global sample_img_path, sample_pdf2img_path, sample_cut_img_path, sample_edited_img_path, sample_sign_img_path
    global debug_mode, debug_font

    input_dir = config_options["PASSPORT_IMAGES_FOLDER_PATH"]
    output_dir = config_options["OUTPUT_FOLDER_PATH"]

    sample_img_path = f"{input_dir}/{passport.file_name}"
    sample_pdf2img_path = f"{output_dir}/{passport.pdf2png_file_name}"
    sample_cut_img_path = f"{output_dir}/{passport.cut_file_name}"
    sample_edited_img_path = f"{output_dir}/{passport.edited_file_name}"
    sample_sign_img_path = f"{output_dir}/{passport.sign_file_name}"

    debug_font = config_options["DEBUG_FONT"]

    # 设置tessract入口程序安装位置
    set_tessract_app()

    # 是否进入调试模式
    if config_options["DEBUG"].lower() == "true":
        debug_mode = True
    elif config_options["DEBUG"].lower() == "false":
        debug_mode = False
    else:
        print("ocr_configs.ini Debug value is ERROR!")


def ocr_by_key(img, key, lang="num_1"):
    # ツール取得
    tool = pyocr.get_available_tools()[0]

    # OCR
    builder = None

    if key == "line":
        builder = pyocr.builders.LineBoxBuilder(tesseract_layout=6)

    if key == "word":
        builder = pyocr.builders.WordBoxBuilder(tesseract_layout=6)

    if lang == "num_1":
        builder.tesseract_configs.append("_my_word")

    data_list = tool.image_to_string(Image.fromarray(img), lang=lang, builder=builder)

    return data_list


# 标记识别结果，并显示图片
def rect_set(img, data_list):
    i = 0
    for data in data_list:
        rect = data.position
        text = data.content

        if debug_mode:
            print(f"{i}{text}   {rect}")

        i += 1

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
        font_path = debug_font
        font_size = 16
        font_color = (0, 0, 255)

        # 加载日语字体
        font = ImageFont.truetype(font_path, font_size)

        # 在图像上绘制文本
        draw.text((x1, y1 - 23), text, font=font, fill=font_color)

    # 将图像从Pillow格式转换为OpenCV格式
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return img_cv


def get_keywords(data_list):

    passport_rect=None
    # 找到 PASSPORT 
    for data in data_list:
        rect = data.position
        text = data.content
        if text.strip()=="PASSPORT":
            passport_rect = rect
            print(data)
            break

    if passport_rect==None:
        raise ValueError('PASSPORT关键字识别失败')

    max_data=("",None)
    for data in data_list:
        rect = data.position
        text = data.content

        if len(text)>len(max_data[0]):
            max_data=(text,rect)
    
    print(f"max_data: {max_data}")
    ret = {
        "passport":passport_rect,
        "max_data":max_data
    }

    return ret

def find_passport(passport: Passport, _config_options: dict):
    global config_options

    config_options = _config_options

    # 初始化设置
    init(passport)

    # 裁切30px不需要的部分,返回处理后的
    img = cv2.imread(sample_pdf2img_path)
    del_img = crop_image(img, 30)
    gray = cv2.cvtColor(del_img, cv2.COLOR_BGR2GRAY)
    thresh = color_scale_display(gray, 112, 217, 0.97)

    x1, y1, x2, y2 = calc_row_and_col_sum(thresh)

    # 使用切片操作截取图像的中心部分top:bottom, left:right
    cut_img = cv2.cvtColor(thresh[int(y1 + (y2 - y1) / 2) : y2, x1:x2], cv2.COLOR_GRAY2BGR)

    data_list = ocr_by_key(cut_img,'word','jpn')
    get_keywords(data_list)

    img_cv = rect_set(cut_img, data_list)
    cv2.imwrite(sample_cut_img_path, img_cv)


# 参数初始化
# 删除空白区域
# 清楚无用信息
