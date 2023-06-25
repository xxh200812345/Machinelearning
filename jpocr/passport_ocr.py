#!/bin/bash
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import pyocr
import pyocr.builders
import cv2
import platform
from collections import Counter

from PIL import Image
import matplotlib.pyplot as plt

import json
from passport import Passport
from datetime import datetime
from collections import Counter

import pytesseract
import os
import re
import random

os.chdir(os.path.abspath(os.path.dirname(__file__)))


# # 返回非空白区域范围
# def calc_row_and_col_sum(img):
#     # 计算每行的值
#     row_sum = np.sum(img, axis=1)

#     # 找到出现次数最多的值
#     max_val = np.max(row_sum)

#     # 行值数组减去出现次数最多的值
#     row_sum = max_val - row_sum

#     # 创建一个布尔数组，指示哪些元素大于 99999
#     mask = row_sum > 9999

#     # 使用布尔索引和 np.where() 函数找到满足条件的行号
#     row_numbers = np.where(mask)[0]

#     # 计算每列的值
#     col_sum = np.sum(img, axis=0)

#     # 找到出现次数最多的值
#     max_val = np.max(col_sum)

#     # 列值数组减去出现次数最多的值
#     col_sum = max_val - col_sum

#     # 创建一个布尔数组，指示哪些元素大于 8888
#     mask = col_sum > 9999

#     # 使用布尔索引和 np.where() 函数找到满足条件的列号
#     col_numbers = np.where(mask)[0]

#     # 返回矩形范围
#     return (
#         col_numbers.min(),
#         row_numbers.min(),
#         col_numbers.max(),
#         row_numbers.max() + 20,
#     )


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

# 设置tessract入口程序安装位置
def set_tessract_app():
    # 获取操作系统名称及版本号
    os = platform.system()
    pyocr.tesseract.TESSERACT_CMD = config_options["WINDOWS_TESSRACT_LOCATION"]

    # 判断当前操作系统
    if os == "Darwin":
        pyocr.tesseract.TESSERACT_CMD = config_options["MAC_TESSRACT_LOCATION"]


def convert_mac_path_to_windows(mac_path):
    """
    将Mac上的相对路径转换为Windows上的相对路径
    """
    windows_path = mac_path.replace("/", "\\")  # 将正斜杠替换为反斜杠
    if windows_path.startswith("\\"):  # 如果路径以根目录开始
        windows_path = windows_path[1:]  # 移除开头的反斜杠
        drive = os.path.splitdrive(os.getcwd())[0]  # 获取当前工作目录的盘符
        windows_path = drive + windows_path  # 添加盘符
    return windows_path


# 初始化设置
def init(passport: Passport):
    global sample_img_path, sample_cut_img_path, sample_edited_img_path, sample_sign_img_path
    global debug_mode, debug_font

    input_dir = config_options["PASSPORT_IMAGES_FOLDER_PATH"]
    output_dir = config_options["OUTPUT_FOLDER_PATH"]

    if passport.ext == ".pdf":
        sample_img_path = (
            f"{output_dir}/{passport.image_dir}/{passport.pdf2png_file_name}"
        )
    else:
        sample_img_path = f"{input_dir}/{passport.image_dir}/{passport.file_name}"

    sample_cut_img_path = f"{output_dir}/{passport.image_dir}/{passport.cut_file_name}"
    sample_edited_img_path = (
        f"{output_dir}/{passport.image_dir}/{passport.edited_file_name}"
    )
    sample_sign_img_path = (
        f"{output_dir}/{passport.image_dir}/{passport.sign_file_name}"
    )

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


def get_mask_rect(words, lines, img):
    """
    获取遮罩范围
    words是单词为的ORC识别
    lines是句子的ORC识别
    """
    # 获取图片的宽度和高度
    height, width = img.shape[:2]

    passport_rect = None
    # 找到 PASSPORT
    for data in words:
        rect = data.position
        text = data.content

        if rect[0][0] > width / 2 or rect[0][1] > height / 2:
            continue

        pattern = r"PASSP[O0o]RT"
        match = re.match(pattern, text.strip().upper())
        if match:
            passport_rect = rect
            print(data)
            break

    if passport_rect == None:
        raise ValueError("PASSPORT关键字识别失败")

    # 和PASSPORT最接近的字符串
    near_datas = []
    for data in words:
        rect = data.position
        text = data.content

        # 提取矩形的上边界和下边界位置
        rect_top = rect[0][1]
        rect_bottom = rect[1][1]  # 下边界位置
        passport_rect_top = passport_rect[0][1]
        passport_rect_bottom = passport_rect[1][1]  # 上边界 + 高度

        rect_h = rect_bottom - rect_top
        passport_rect_h = passport_rect_bottom - passport_rect_top
        max_h = max(
            abs(rect_top - passport_rect_bottom), abs(rect_bottom - passport_rect_top)
        )

        if "JPN" in text:
            print(text)

        # 判断两个矩形是否在垂直方向上重叠，且在PASSPORT的后面
        if max_h < rect_h + passport_rect_h and passport_rect[1][0] < rect[1][0]:
            near_datas.append(data)

    # mrz
    dict_array = []
    for data in lines:
        rect = data.position
        text = data.content

        # 排除靠上的数据
        if rect[0][1] < height * 0.70:
            continue

        dict_array.append({"text": text, "data": data})

    sorted_array = sorted(dict_array, key=lambda x: len(x["text"]))
    mrz2 = sorted_array[-2]["data"]
    mrz1 = sorted_array[-1]["data"]

    print(f"mrz1: {mrz1.content},{mrz1.position}")
    print(f"mrz2: {mrz2.content},{mrz2.position}")

    p_rect = None
    jpn_rect = None
    passportno_rect = None
    for data in near_datas:
        rect = data.position
        text = data.content

        # if debug_mode:
        print(f"{text}: {rect}")

        if "p" in text.strip().lower() and len(text) < 3:
            p_rect = data.position
            continue

        if "jpn" in text.strip().lower() and len(text) < 5:
            jpn_rect = data.position
            continue

        pattern = r"^[a-zA-Z]{0,3}\d{5,9}$"
        match = re.match(pattern, text.strip().lower())
        if match:
            passportno_rect = data.position

    if passportno_rect == None:
        raise ValueError("Passport No关键字识别失败")

    if p_rect == None:
        ValueError("P关键字识别失败, 用jpn继续定位")
        if jpn_rect == None:
            raise ValueError("jpn关键字识别失败")
        else:
            # x1,y1 x2,y2
            ((x1, y1), (x2, y2)) = jpn_rect
            word_width = int((x2 - x1) / 3)
            p_rect = (
                (x1 - (passportno_rect[0][0] - x2) - word_width, y1),
                (x1 - (passportno_rect[0][0] - x2), y2),
            )

    # 创建一个全白色的遮罩，它的大小和原图一样
    mask = np.ones((height, width), dtype="uint8") * 255

    # 定义多边形顶点坐标
    mrz2_rect = mrz2.position
    mrz1_rect = mrz1.position

    broder = 10

    mrz2_y2 = mrz2_rect[1][1] + broder if mrz2_rect[1][1] + broder < height else height
    mrz1_x1 = mrz1_rect[0][0] - broder if mrz1_rect[0][0] - broder > 0 else 0

    points = np.array(
        [
            [p_rect[0][0] - broder, passportno_rect[0][1] - broder],  # 左上
            [passportno_rect[1][0] + broder, passportno_rect[0][1] - broder],
            [mrz2_rect[1][0] + broder, passportno_rect[0][1] - broder],  # 右上
            [mrz2_rect[1][0] + broder, mrz2_y2],  # 右下
            [mrz1_x1, mrz2_y2],  # 左下
            [mrz1_x1, mrz1_rect[0][1] - broder],
            [p_rect[0][0] - broder, mrz1_rect[0][1] - broder],
        ],
        dtype=np.int32,
    )
    # 重新调整维度
    pts = points.reshape((-1, 1, 2))

    # 在mask上绘制多边形
    cv2.fillPoly(mask, [pts], (0))

    # 把mask和原图进行“与”操作，得到遮罩部分的图像
    res = cv2.bitwise_or(img, mask)

    border = 2

    y1 = passportno_rect[0][1] - border if passportno_rect[0][1] - border > 0 else 0
    y2 = mrz2_rect[1][1] + border if mrz2_rect[1][1] + border < height else height
    x1 = mrz2_rect[0][0] - border if mrz2_rect[0][0] - border > 0 else 0
    x2 = mrz2_rect[1][0] + border if mrz2_rect[1][0] + border < width else width

    res = res[
        y1:y2,
        x1:x2,
    ]

    print(f"有效数据图片大小：{res.shape}")

    border = int(height * 0.05)
    res = add_border_to_grayscale_image(res, border)

    return res


# 白底灰度图像边框加宽
def add_border_to_grayscale_image(image, border_size=10, border_color=255):
    # 获取图像的尺寸
    image_height, image_width = image.shape

    # 计算背景的尺寸
    background_height = image_height + (2 * border_size)
    background_width = image_width + (2 * border_size)

    # 创建背景图像
    background = np.full(
        (background_height, background_width), border_color, dtype=np.uint8
    )

    # 将图像放置在背景中心
    x = (background_width - image_width) // 2
    y = (background_height - image_height) // 2
    background[y : y + image_height, x : x + image_width] = image

    return background


# 基于哈希的字符串相似度检测
def minhash_similarity(str1, str2, num_hashes=100):
    # 定义哈希函数
    def hash_function(x):
        return hash(x)

    # 生成哈希签名
    def generate_minhash_signature(string, num_hashes):
        signature = []
        for i in range(num_hashes):
            min_hash = float("inf")
            for token in string:
                hash_value = hash_function(str(i) + token)
                if hash_value < min_hash:
                    min_hash = hash_value
            signature.append(min_hash)
        return signature

    # 计算相似度
    def calculate_similarity(sig1, sig2):
        intersection = len(set(sig1) & set(sig2))
        union = len(set(sig1) | set(sig2))
        similarity = intersection / union
        return similarity

    # 生成字符串的哈希签名
    signature1 = generate_minhash_signature(str1, num_hashes)
    signature2 = generate_minhash_signature(str2, num_hashes)

    # 计算相似度
    similarity = calculate_similarity(signature1, signature2)
    return similarity


# 获取数组中出现次数最多的值
def get_most_common_elements(array):
    counter = Counter(array)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None


# 获取识别文字的位置信息
def get_text_location(image):
    try:
        data = pytesseract.image_to_osd(image)
    except pytesseract.pytesseract.TesseractError:
        return None
    # print(data)
    # 从位置信息中提取文字方向
    lines = data.split("\n")
    angle = None
    for line in lines:
        if line.startswith("Orientation in degrees:"):
            angle = float(line.split(":")[1].strip())
            break

    return angle


# 获取识别文字的方向，并旋转图片，只能识别90 180 270
def rotate_image_with_white_bg(image):
    # 获取识别文字的位置信息
    angle = []
    data_list_word = ocr_by_key(image, "line", "jpn")

    # 获取图像尺寸
    height, width = image.shape[:2]

    # 使用列表推导式筛选出长度大于 5 的对象
    _data_list_word = [data for data in data_list_word if len(data.content) > 5]

    for data in random.sample(_data_list_word, 20):
        rect = data.position

        # 裁剪图像
        border = 20
        ((x1, y1), (x2, y2)) = rect
        x1 = 0 if x1 - border < 0 else x1 - border
        y1 = 0 if y1 - border < 0 else y1 - border
        x2 = width if x2 + border > width else x2 + border
        y2 = height if y2 + border > height else y2 + border
        cropped_image = image[y1:y2, x1:x2]
        # _imshow("",cropped_image)
        item_angle = get_text_location(cropped_image)
        if item_angle is not None:
            angle.append(item_angle)

    angle = get_most_common_elements(angle)

    # 计算旋转中心点
    center = (width // 2, height // 2)

    # 构建旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算旋转后图像的尺寸
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # 调整旋转矩阵中心点偏移量
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # 执行旋转变换并填充白色背景
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (new_width, new_height), borderValue=(255, 255, 255)
    )

    return rotated_image


def _imshow(title, img, scale_percent=50):
    # 缩小比例
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(img, (width, height))

    # if debug_mode == True:
    cv2.imshow(title, resized_image)
    cv2.waitKey(0)  # 等待用户按下键盘上的任意键
    cv2.destroyAllWindows()  # 关闭所有cv2.imshow窗口


# 只返回指定高度以内的区域（max，min）
def remove_small_height_regions(mask, img, max_height, min_height):
    # 对输入图像取反
    inverted_img = cv2.bitwise_not(img)
    # 膨胀操作
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 1))
    bin_clo = cv2.dilate(inverted_img, kernel2, iterations=2)
    # _imshow("Gaussian Thresholding", bin_clo)

    # 获取所有连通区域的标签
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_clo, connectivity=8
    )

    # 遍历每个连通区域，计算它们的高度
    for i in range(1, num_labels):
        height = stats[i, cv2.CC_STAT_HEIGHT]
        x1 = stats[i, cv2.CC_STAT_LEFT]
        y1 = stats[i, cv2.CC_STAT_TOP]
        x2 = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]
        y2 = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
        # 高度设置大于指定值的区域
        if height > min_height and height < max_height:
            cv2.rectangle(mask, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), 255, -1)

    return mask


def resize_image_by_height(image, target_height):
    # 获取原始图像的尺寸
    height, width = image.shape[:2]

    # 计算缩放因子
    scale_factor = target_height / height

    # 计算缩放后的宽度
    target_width = int(width * scale_factor)

    # 缩放图像
    resized_image = cv2.resize(image, (target_width, target_height))

    return resized_image


def mask_fill_white(img, mask):
    # 将矩形取反
    mask_inv = cv2.bitwise_not(mask)
    img_fg = cv2.bitwise_and(img, img, mask=mask_inv)

    # 创建一个白色背景
    background = np.ones(img.shape, dtype=np.uint8) * 255
    result = cv2.bitwise_and(background, background, mask=mask)

    # 将矩形内部区域与矩形外部区域合并
    result = cv2.add(img_fg, result)

    return result


# 图片二值化，把白色部分设置为透明
def binary_img_with_transparency(img, threshold=180):
    # 将二值化后的图像转换为4通道图像
    rgba = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)

    # 将白色部分设置为透明
    rgba[:, :, 3] = np.where(img == 255, 0, 255)
    return rgba


# 识别后数据输出到文本文件中
def output_data2text_file(passport_list, _config_options: dict):
    for passport in passport_list:
        output_data_file = (
            _config_options["OUTPUT_FOLDER_PATH"] + "/" + passport.file_name + ".json"
        )
        # 打开文件，将文件指针移动到文件的末尾
        with open(output_data_file, "a", encoding="utf-8") as f:
            json.dump(passport.info, f, ensure_ascii=False)


def is_point_in_rect(point, rect):
    """
    判断点是否在矩形区域内

    参数:
    - point: 一个包含两个元素的元组或列表，表示点的坐标 (x, y)
    - rect: 一个包含四个元素的元组或列表，表示矩形的坐标 (x1, y1, x2, y2)

    返回值:
    - 如果点在矩形区域内，则返回 True，否则返回 False
    """
    (x, y) = point
    ((x1, y1), (x2, y2)) = rect

    if x1 <= x <= x2 and y1 <= y <= y2:
        return True
    else:
        return False


def check_len(ret):
    for title, key_len in Passport.PASSPORT_KEYS_LEN.items():
        if key_len > 0 and len(ret[title]) < key_len:
            ret["vs_info"][
                title
            ] = f"{Passport.OUT_ERROR_TAG}: 实际长度{len(ret[title])}小于预测长度{key_len}"
        else:
            ret["vs_info"][title] = ""


def to_O(text):
    return text.replace("0", "O")


def to_0(text):
    return text.replace("O", "0")


def set_main_info(ret):
    main_info = ret["main_info"]
    vs_info = ret["vs_info"]

    main_info[Passport.Type] = ret[Passport.Type]
    main_info[Passport.Issuing_country] = to_O(ret[Passport.Issuing_country])

    tmp = Passport.Passport_No
    if vs_info[tmp][:5] != Passport.OUT_ERROR_TAG:
        main_info[tmp] = to_O(ret[tmp][:2]) + to_0(ret[tmp][2:])

    main_info[Passport.Surname] = to_O(ret[Passport.Surname])
    main_info[Passport.Given_name] = to_O(ret[Passport.Given_name])
    main_info[Passport.Nationality] = to_O(ret[Passport.Nationality])

    tmp = Passport.Date_of_birth
    if vs_info[tmp][:5] != Passport.OUT_ERROR_TAG:
        main_info[tmp] = to_0(ret[tmp][:2]) + to_O(ret[tmp][2:5]) + to_0(ret[tmp][-4:])

    main_info[Passport.Sex] = ret[Passport.Sex]

    main_info[Passport.Registered_Domicile] = to_O(ret[Passport.Registered_Domicile])

    tmp = Passport.Date_of_issue
    if vs_info[tmp][:5] != Passport.OUT_ERROR_TAG:
        main_info[tmp] = to_0(ret[tmp][:2]) + to_O(ret[tmp][2:5]) + to_0(ret[tmp][-4:])

    tmp = Passport.Date_of_expiry
    if vs_info[tmp][:5] != Passport.OUT_ERROR_TAG:
        main_info[tmp] = to_0(ret[tmp][:2]) + to_O(ret[tmp][2:5]) + to_0(ret[tmp][-4:])


def set_mrz_info(ret):
    mrz_info = ret["mrz_info"]
    vs_info = ret["vs_info"]
    foot1 = to_O(ret[Passport.foot1])
    foot2 = ret[Passport.foot2]

    if vs_info[Passport.foot1][:5] != Passport.OUT_ERROR_TAG:
        Surname_p2 = 0
        Given_name_p2 = 0
        for title, position in Passport.PASSPORT_MRZ1_POSITION.items():
            if title == Passport.Surname:
                find_surname = foot1[position[0] :]
                Surname_p2 = find_surname.find("<")
                if Surname_p2 != -1:
                    mrz_info[title] = find_surname[:Surname_p2]
                    Surname_p2 = position[0] + len(mrz_info[title])
                else:
                    mrz_info[title] = Passport.OUT_ERROR_TAG + ": 找不到结尾<"

            elif title == Passport.Given_name:
                if Surname_p2 + 2 > len(foot1):
                    mrz_info[title] = Passport.OUT_ERROR_TAG + ": 超过了MRZ的长度"
                elif mrz_info[Passport.Surname][:5] == Passport.OUT_ERROR_TAG:
                    mrz_info[title] = Passport.OUT_ERROR_TAG + ": 姓没找到，所以放弃找名"
                else:
                    find_given_name = foot1[Surname_p2 + 2 :]
                    Given_name_p2 = find_given_name.find("<")
                    if Given_name_p2 != -1:
                        mrz_info[title] = find_given_name[:Given_name_p2]
                    else:
                        mrz_info[title] = Passport.OUT_ERROR_TAG + ": 找不到结尾<"

            else:
                mrz_info[title] = foot1[position[0] : position[1]]
    else:
        for title, position in Passport.PASSPORT_MRZ1_POSITION.items():
            mrz_info[title] = ""

    if vs_info[Passport.foot2][:5] != Passport.OUT_ERROR_TAG:
        for title, position in Passport.PASSPORT_MRZ2_POSITION.items():
            tmp = foot2[position[0] : position[1]]

            if title == Passport.Passport_No:
                mrz_info[title] = to_O(tmp[:2]) + to_0(tmp[2:])

            if title == Passport.Nationality:
                mrz_info[title] = to_O(tmp)

            if title == Passport.Date_of_birth:
                mrz_info[title] = to_0(tmp)

            if title == Passport.Date_of_expiry:
                mrz_info[title] = to_0(tmp)

            mrz_info[title] = tmp
    else:
        for title, position in Passport.PASSPORT_MRZ1_POSITION.items():
            mrz_info[title] = ""


def get_month_number(abbreviation):
    """
    将3个字母的月份缩写转换为对应的月份数字
    """
    try:
        date_object = datetime.strptime(abbreviation, "%b")
        month_number = date_object.month
        return month_number
    except ValueError:
        return f"月份{abbreviation}转换错误"


def add_error_to_info(info, error_msg):
    """
    追加对应的比较错误 info：vs_info[title]
    """
    if info[:5] != Passport.OUT_ERROR_TAG:
        info = Passport.OUT_ERROR_TAG + ": " + error_msg
    else:
        info += ";" + error_msg

    return info


def clear_little_px(image):
    """
    清除独立小像素点
    """

    # 二值化处理
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制矩形轮廓
    new_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < 400 and h < 20:
            new_contours.append(contour)

    # 创建与原始图像相同大小的空白掩码
    mask = np.zeros_like(image)

    # 填充轮廓区域为白色
    cv2.drawContours(mask, new_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    result = cv2.bitwise_or(image, mask)

    return result


def set_vs_info(ret):
    main_info = ret["main_info"]
    mrz_info = ret["mrz_info"]
    vs_info = ret["vs_info"]

    for title, mrz_item in mrz_info.items():
        if hasattr(main_info, title) == False:
            main_info[title] = f"main_info中不存在这个属性"
            continue

        if main_info[title][:5] == Passport.OUT_ERROR_TAG:
            error_msg = f"中间的信息项目存在错误"
            vs_info[title] = add_error_to_info(vs_info[title], error_msg)

        elif mrz_info[title][:5] == Passport.OUT_ERROR_TAG:
            error_msg = f"MRZ的信息项目存在错误"
            vs_info[title] = add_error_to_info(vs_info[title], error_msg)

        elif main_info == "":
            error_msg = f"中间的信息项目没有值"
            vs_info[title] = add_error_to_info(vs_info[title], error_msg)

        elif mrz_info == "":
            error_msg = f"MRZ的信息项目没有值"
            vs_info[title] = add_error_to_info(vs_info[title], error_msg)

        elif title == Passport.Date_of_birth or title == Passport.Date_of_expiry:
            month_num = get_month_number(main_info[title][2:5])
            if isinstance(month_num, str):
                error_msg = month_num
                vs_info[title] = add_error_to_info(vs_info[title], error_msg)
            else:
                main_item = (
                    main_info[title][-2:]
                    + str(month_num).zfill(2)
                    + main_info[title][:2]
                )
                if main_item != mrz_item:
                    error_msg = f"数据不一致"
                    vs_info[title] = add_error_to_info(vs_info[title], error_msg)

        elif title == Passport.Nationality:
            nationality_code = Passport.get_nationality_code(main_info[title])
            if nationality_code == "UNK":
                error_msg = f"没找到对应的国家（{main_info[title]}）code"
                vs_info[title] = add_error_to_info(vs_info[title], error_msg)

            elif nationality_code != mrz_info[title]:
                error_msg = f"数据不一致"
                vs_info[title] = add_error_to_info(vs_info[title], error_msg)

        elif main_info[title] != mrz_info[title]:
            error_msg = f"数据不一致"
            vs_info[title] = add_error_to_info(vs_info[title], error_msg)

    return


# 获取护照信息
def datalist2info(passport: Passport, data_list):
    ret = passport.info
    ret["main_info"] = {}
    ret["mrz_info"] = {}
    ret["vs_info"] = {}

    ocr_texts = ""
    for data in data_list:
        # 文本
        ocr_texts += f"{data.content} {data.position}\n"

    ret["file_name"] = passport.file_name

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ret["time"] = now_str

    if len(data_list) != 13:
        err_msg = "识别后文字信息行数不为13，有识别错误。"
        ret["err_msg"] = add_error_to_info(ret["err_msg"], err_msg)

    ret["ocr_texts"] = ocr_texts

    error_vals = 0
    for key, value in Passport.PASSPORT_KEYS_POSITION.items():
        for data in data_list:
            rect = data.position
            text = data.content
            if is_point_in_rect(value, rect):
                ret[key] = text
                break

        if key not in ret:
            ret[key] = Passport.OUT_ERROR_TAG + ": 没有找到数据."
            error_vals += 1

    if error_vals > 0:
        err_msg = f"一共有{error_vals}个数据没有找到对应值。"
        ret["err_msg"] = add_error_to_info(ret["err_msg"], err_msg)

    # 根据基础信息生成三个对象，对象main_info保存护照主要信息，对象mrz_info保存下方mrz分解后的信息，对象vs_info保存对比信息
    check_len(ret)
    set_main_info(ret)
    set_mrz_info(ret)
    set_vs_info(ret)

    return ret


def fill_middle_with_white(image, style):
    """
    把图片的上下部分涂白
    """

    # 获取图像的尺寸
    height, width = image.shape[:2]

    if style == "下边涂白":
        # 计算矩形区域的左上角和右下角坐标
        top_left = (0, int(height * 0.6))
        bottom_right = (width, height)
    else:
        # 计算矩形区域的左上角和右下角坐标
        top_left = (0, 0)
        bottom_right = (width, int(height * 0.4))

    # 创建与原始图像大小相同的空白图像
    mask = np.zeros_like(image)

    # 在空白图像上绘制矩形区域为白色
    cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255), cv2.FILLED)

    # 将矩形区域应用到原始图像上
    result = cv2.bitwise_or(image, mask)

    return result


def test(passport: Passport):
    # # 使用PIL库打开图像文件
    # pil_image = Image.open(sample_img_path)
    # img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh = color_scale_display(gray, 112, 217, 0.97)

    # # 获取识别文字的方向，并旋转图片，只能识别90 180 270
    # thresh = rotate_image_with_white_bg(thresh)

    # 使用PIL库打开图像文件
    pil_image = Image.open("to_rotate" + "\\" + passport.file_name + "_rotate.png")
    thresh = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)

    # MZR
    data_list_line = ocr_by_key(thresh, "line", "jpn")

    data_list = [data for data in data_list_line if len(data.content) > 5]

    mrz_datas = []
    for data in data_list:
        text = data.content

        pattern = r".*[<くぐ]{2}.*[<くぐ]{2}.*"
        match = re.match(pattern, text.strip().upper())
        if match:
            mrz_datas.append(data)

    for data in mrz_datas:
        print(f"{data.content},{data.position}")

    data_list_line = ocr_by_key(thresh, "line", "jpn")

    mrz_1 = mrz_datas[0].position
    mrz_2 = mrz_datas[1].position

    mrz_h = mrz_2[1][1] - mrz_1[0][1]
    mrz_t = mrz_1
    if mrz_t[0][0] < mrz_2[0][0]:
        mrz_t = mrz_2

    # 获得图片
    thresh = thresh[
        mrz_2[1][1] - mrz_h * 8 : mrz_2[1][1], mrz_t[0][0] - 5 : mrz_t[1][0] + 5
    ]

    # 存储OCR结果图片
    data_list_line = ocr_by_key(thresh, "line", "jpn")
    img_cv = rect_set(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), data_list_line)

    plt.imsave(
        "to_init" + "\\" + passport.file_name + "_init.png",
        cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
    )
    plt.imsave("to_init" + "\\" + passport.file_name + "_init_ocr.png", img_cv)


def main(passport: Passport, _config_options: dict):
    global config_options

    config_options = _config_options

    # 初始化设置
    init(passport)

    # 使用PIL库打开图像文件
    pil_image = Image.open(sample_img_path)
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = color_scale_display(gray, 112, 217, 0.97)

    # 获取识别文字的方向，并旋转图片，只能识别90 180 270
    thresh = rotate_image_with_white_bg(thresh)

    # pil_image = Image.open("to_rotate" + "\\" + passport.file_name + "_rotate.png")
    # thresh = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)

    data_list_line = ocr_by_key(thresh, "line", "jpn")
    data_list = [data for data in data_list_line if len(data.content) > 5]

    # MZR
    mrz_datas = []
    for data in data_list:
        text = data.content

        pattern = r".*[<くぐ]{2}.*[<くぐ]{2}.*"
        match = re.match(pattern, text.strip().upper())
        if match:
            mrz_datas.append(data)

    for data in mrz_datas:
        print(f"{data.content},{data.position}")

    # passport
    passport_data = None
    passport_cnt = 0
    patterns = [
        r".*旅券 .*",
        r".* PASSP[O0]RT .*",
        r".* .?P .*",
        r".* JPN .*",
        r".* [a-zA-Z]{1,3}\d{6,8}$ .*",
    ]
    for data in data_list:
        text = data.content
        passport_cnt = 0

        for pattern in patterns:
            match = re.match(pattern, text.strip().upper())
            if match:
                passport_cnt += 1

        if passport_cnt >= 2:
            passport_data = data
            break

    if passport_data:
        print(
            f"passport行命中率：{round(passport_cnt/len(patterns),2)},{passport_data.content}"
        )
    else:
        print(f"passport行未命中率：{round(passport_cnt/len(patterns),2)}")

    data_list_line = ocr_by_key(thresh, "line", "jpn")

    mrz_1 = mrz_datas[0].position
    mrz_2 = mrz_datas[1].position

    mrz_t = mrz_1
    if mrz_t[0][0] < mrz_2[0][0]:
        mrz_t = mrz_2

    # 获得图片
    if passport_data:
        y1 = passport_data.position[0][1]- 5 
    else:
        y1 = mrz_2[1][1] - (mrz_2[1][1] - mrz_2[0][1]) * 25

    thresh = thresh[
        y1: mrz_2[1][1],
        mrz_t[0][0] - 5 : mrz_t[1][0] + 5,
    ]

    thresh_OCR = fill_middle_with_white(thresh.copy(), "下边涂白")

    data_list_word = ocr_by_key(thresh_OCR, "word", "jpn")

    data_list_line = ocr_by_key(thresh, "line", "jpn")

    cut_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    img_cv = rect_set(cut_img, data_list_word)

    plt.imsave(sample_cut_img_path, img_cv)

    # 获取数据范围
    thresh = get_mask_rect(data_list_word, data_list_line, thresh)
    # 缩放到固定高度600
    thresh = resize_image_by_height(thresh, 800)

    # 二值化图像
    _, img_mask = cv2.threshold(thresh, 150, 255, cv2.THRESH_BINARY)
    height, width = img_mask.shape[:2]
    # 读取原始图像和矩形遮罩
    mask = np.zeros(img_mask.shape[:2], dtype=np.uint8)

    # 去除签名 top:bottom, left:right
    x1, y1, x2, y2 = (
        int(width * 0.585),
        int(height * 0.52),
        width - 1,
        int(height * 0.75),
    )
    sign_rect = ((x1, y1), (x2, y2))
    cv2.rectangle(mask, sign_rect[0], sign_rect[1], 255, -1)
    plt.imsave(
        sample_sign_img_path, binary_img_with_transparency(img_mask[y1:y2, x1:x2])
    )

    # 去除水印
    x1, y1, x2, y2 = (
        int(width * 0.75),
        int(height * 0.25),
        width - 1,
        int(height * 0.52),
    )
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # 去除title top:bottom, left:right
    mask = remove_small_height_regions(mask, img_mask, 20, 3)

    # 遮罩外涂白
    img = mask_fill_white(thresh, mask)
    img = clear_little_px(img)
    img = add_border_to_grayscale_image(img,100)

    # OCR
    data_list = ocr_by_key(img, "word", "num_1")

    dict_array = []
    top_data = None
    for data in data_list:
        rect = data.position
        text = data.content

        if top_data :
            if top_data.position[0][1] > rect[0][1]:
                top_data = data
        else:
            top_data = data

        # 排除靠上的数据
        if rect[0][1] < height * 0.70:
            continue

        dict_array.append({"text": text, "data": data})

    sorted_array = sorted(dict_array, key=lambda x: len(x["text"]))
    mrz2 = sorted_array[-2]["data"]
    mrz1 = sorted_array[-1]["data"]

    y1 = top_data.position[0][1] - 62
    x1 = min(mrz1.position[0][0],mrz2.position[0][0]) - 54

    class Ocr_ret:
        def __init__(self, position, content):
            self.position = position
            self.content = content

    _data_list = []
    for data in data_list:
        rect = data.position
        text = data.content

        _rect = ((rect[0][0] - x1,rect[0][1] - y1),(rect[1][0] - x1,rect[1][1] - y1))

        _data_list.append(Ocr_ret(_rect,text))

    height, width = img.shape[:2]
    img = img[y1:height,x1:width]

    cut_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 存储OCR结果图片
    img_cv = rect_set(cut_img, _data_list)
    # img_cv = cut_img
    plt.imsave(sample_edited_img_path, img_cv)

    # 获取护照信息
    passport.info = datalist2info(passport, _data_list)

def run(passport: Passport, _config_options: dict):
    try:
        main(passport, _config_options)
    except Exception as e:
        # 捕获异常并打印错误信息
        print(f"发生错误 {passport.file_name}:", str(e))

        ret = passport.info
        ret["err_msg"] = add_error_to_info(ret["err_msg"], str(e))
