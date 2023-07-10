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

from passport import Passport
from datetime import datetime
from collections import Counter

import pytesseract
import os
import re
import difflib
import json

os.chdir(os.path.abspath(os.path.dirname(__file__)))


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
    """
    key : line word
    lang : jpn eng num_1
    """
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
def rect_set(img, data_list, color=(0, 0, 255)):
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
        cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), color, 2)

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
        font_color = color

        # 加载日语字体
        font = ImageFont.truetype(font_path, font_size)

        # 在图像上绘制文本
        draw.text((x1, y1 - 23), text, font=font, fill=font_color)

    # 将图像从Pillow格式转换为OpenCV格式
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return img_cv


def merge_rectangles(rect1, rect2):
    """
    合并矩形
    """
    x1 = min(rect1[0][0], rect2[0][0])
    y1 = min(rect1[0][1], rect2[0][1])
    x2 = max(rect1[1][0], rect2[1][0])
    y2 = max(rect1[1][1], rect2[1][1])
    merged_rect = ((x1, y1), (x2, y2))
    return merged_rect


def check_similarity(string, pattern, similarity_threshold=0.6):
    """
    相似度规则，并使用正则表达式模式进行匹配
    """
    similarity = difflib.SequenceMatcher(None, string, pattern).ratio()
    return similarity >= similarity_threshold


def find_mask_key_point_by_passport(passport_rect, words, img):
    # 获取图片的宽度和高度
    height, width = img.shape[:2]

    # 和PASSPORT最接近的字符串
    near_datas = []
    for data in words:
        rect = data.position
        text = data.content

        # 提取矩形的上边界和下边界位置
        rect_top = rect[0][1]
        rect_bottom = rect[1][1]  # 下边界位置
        passport_rect_top = passport_rect[0][1] - 10  # 存在passportno在非常上面的情况 10px
        passport_rect_bottom = passport_rect[1][1]  # 上边界 + 高度

        rect_h = rect_bottom - rect_top
        passport_rect_h = passport_rect_bottom - passport_rect_top
        max_h = max(
            abs(rect_top - passport_rect_bottom), abs(rect_bottom - passport_rect_top)
        )

        # 判断两个矩形是否在垂直方向上重叠，且在PASSPORT的后面
        if max_h < rect_h + passport_rect_h and passport_rect[1][0] < rect[1][0]:
            near_datas.append(data)

    p_rect = None
    jpn_rect = None
    passportno_rect = None
    for data in near_datas:
        rect = data.position
        text = data.content

        if debug_mode:
            print(f"{text}: {rect}")

        if "p" in text.strip().lower() and len(text) < 3 and rect[0][0] < width * 0.4:
            p_rect = data.position
            continue

        if "jpn" in text.strip().lower() and len(text) < 5:
            jpn_rect = data.position
            continue

        elif check_similarity(text.strip().lower(), r"jpn"):
            jpn_rect = data.position
            continue

        pattern = r"^[a-zA-Z]{0,3}\d{5,9}$"
        match = re.match(pattern, text.strip().lower())
        if match:
            passportno_rect = data.position

    if passportno_rect == None:
        # 如果检测结果过于分离，尝试合并一下看
        if jpn_rect:
            passportno_data_temps = []
            for data in near_datas:
                rect = data.position
                text = data.content

                if rect[0][0] > jpn_rect[1][0]:
                    passportno_data_temps.append(data)

            sorted_array = sorted(
                passportno_data_temps, key=lambda x: data.position[0][0]
            )
            passportno_text = ""
            passportno_rect_temp = None
            for data in sorted_array:
                rect = data.position
                text = data.content
                passportno_text += text
                if passportno_rect_temp:
                    passportno_rect_temp = merge_rectangles(passportno_rect_temp, rect)
                else:
                    passportno_rect_temp = rect

            pattern = r"^[a-zA-Z]{0,3}\d{5,9}$"
            match = re.match(pattern, passportno_text.strip().lower())
            if match:
                passportno_rect = passportno_rect_temp

    if p_rect == None:
        print("P关键字识别失败, 用jpn继续定位")
        if jpn_rect != None:
            # x1,y1 x2,y2
            ((x1, y1), (x2, y2)) = jpn_rect
            word_width = int((x2 - x1) / 3)
            p_rect = (
                (x1 - (passportno_rect[0][0] - x2) - word_width, y1),
                (x1 - (passportno_rect[0][0] - x2), y2),
            )
        else:
            print("jpn关键字识别失败")

    return p_rect, passportno_rect


def get_mask_rect(words, lines, img):
    """
    获取遮罩范围
    words是单词为的ORC识别
    lines是句子的ORC识别
    """
    # 获取图片的宽度和高度
    height, width = img.shape[:2]

    passport_rect = None
    type_title_rect = None
    kata_type_title_rect = None
    passport_title_rect = None
    no_title_rect = None
    for data in words:
        rect = data.position
        text = data.content.strip().lower()

        if (
            check_similarity(text, r"passport", 0.7)
            and rect[0][0] < width * 0.5
            and rect[0][1] < height * 0.5
        ):
            passport_rect = rect

        if (
            check_similarity(text, r"型", 0.7)
            and len(text) <= 2
            and rect[0][0] < width * 0.8
            and rect[0][0] > width * 0.2
            and rect[0][1] < height * 0.5
        ):
            kata_type_title_rect = rect

        if (
            check_similarity(text, r"type", 0.7)
            and len(text) <= 5
            and rect[0][0] < width * 0.8
            and rect[0][0] > width * 0.2
            and rect[0][1] < height * 0.5
        ):
            type_title_rect = rect

        if (
            check_similarity(text, r"passport", 0.7)
            and rect[0][0] > width * 0.5
            and rect[0][1] < height * 0.5
        ):
            passport_title_rect = rect

        if (
            check_similarity(text, r"no.", 0.7)
            and len(text) <= 4
            and rect[0][0] > width * 0.5
            and rect[0][1] < height * 0.5
        ):
            no_title_rect = rect

    p_rect, passportno_rect = None, None
    if passport_rect:
        p_rect, passportno_rect = find_mask_key_point_by_passport(
            passport_rect, words, img
        )
    else:
        print("PASSPORT关键字识别失败")

    top_left_x = None  # 左上 x
    top_right_1_x = None  # 右上1 x
    top_y = None  # 左上 右上 y

    if not p_rect:
        if kata_type_title_rect:
            ((x1, y1), (x2, y2)) = kata_type_title_rect
            top_left_x = x1 if x1 > 0 else 0  # 左上 x
            print("型 title -> 定位左上 x")

        elif type_title_rect:
            ((x1, y1), (x2, y2)) = kata_type_title_rect
            top_left_x = x1 - int((x2 - x1) * 0.5)
            top_left_x = top_left_x if top_left_x < width else width  # 左上 x
            print("type title -> 定位左上 x")

        else:
            raise ValueError("p关键字区域识别失败")
    else:
        top_left_x = p_rect[0][0]  # 左上 x

    if not passportno_rect:
        if passport_title_rect:
            ((x1, y1), (x2, y2)) = passport_title_rect
            ((x1, y1), (x2, y2)) = ((0, y2 + 10), (x2 + (x2 - x1), height))
            (x1, y1, x2, y2) = rect_vs_box(((x1, y1), (x2, y2)), width, height)
            top_right_1_x = x2  # 右上1 x
            top_y = y1  # 左上 右上 y
            print("passport title -> 定位 passportno 关键字")

        elif no_title_rect:
            ((x1, y1), (x2, y2)) = no_title_rect
            ((x1, y1), (x2, y2)) = ((0, y2 + 10), (x2 + (x2 - x1) * 2, height))
            (x1, y1, x2, y2) = rect_vs_box(((x1, y1), (x2, y2)), width, height)
            top_right_1_x = x2  # 右上1 x
            top_y = y1  # 左上 右上 y
            print("No. title -> 定位 passportno 关键字")

        else:
            raise ValueError("passportno区域关键字识别失败")
    else:
        top_right_1_x = passportno_rect[1][0]  # 右上1 x
        top_y = passportno_rect[0][1]  # 左上 右上 y

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

    if debug_mode:
        print(f"mrz1: {mrz1.content},{mrz1.position}")
        print(f"mrz2: {mrz2.content},{mrz2.position}")

    # 创建一个全白色的遮罩，它的大小和原图一样
    mask = np.ones((height, width), dtype="uint8") * 255

    # 定义多边形顶点坐标
    mrz2_rect = mrz2.position
    mrz1_rect = mrz1.position

    # 最短 mrz
    mrz_s_rect = mrz1_rect
    if mrz1_rect[0][0] < mrz2_rect[0][0]:
        mrz_s_rect = mrz2_rect

    broder = 10

    mrz2_y2 = mrz2_rect[1][1] + broder if mrz2_rect[1][1] + broder < height else height
    mrz1_x1 = mrz_s_rect[0][0] - broder if mrz_s_rect[0][0] - broder > 0 else 0

    top_right_2_x = mrz2_rect[1][0]  # 右下 右上2 x
    mrz1_y = mrz1_rect[0][1]

    points = np.array(
        [
            [top_left_x - broder, top_y - broder],  # 左上
            [top_right_1_x + broder, top_y - broder],  # 右上1
            [top_right_2_x + broder, top_y - broder],  # 右上2
            [top_right_2_x + broder, mrz2_y2],  # 右下
            [mrz1_x1, mrz2_y2],  # 左下
            [mrz1_x1, mrz1_y - broder],
            [top_left_x - broder, mrz1_y - broder],
        ],
        dtype=np.int32,
    )
    # 重新调整维度
    pts = points.reshape((-1, 1, 2))

    # 在mask上绘制多边形
    cv2.fillPoly(mask, [pts], (0))

    # 把mask和原图进行“与”操作，得到遮罩部分的图像
    res = cv2.bitwise_or(img, mask)

    border = 10

    y1 = top_y - border if top_y - border > 0 else 0
    y2 = mrz2_rect[1][1] + border if mrz2_rect[1][1] + border < height else height
    x1 = mrz2_rect[0][0] - border if mrz2_rect[0][0] - border > 0 else 0
    x2 = mrz2_rect[1][0] + border if mrz2_rect[1][0] + border < width else width

    res = res[
        y1:y2,
        x1:x2,
    ]

    if debug_mode:
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


def uniform_sampling(data, sample_size):
    """
    在一个数组中均匀抽取sample_size条数据
    """
    n = len(data)
    if sample_size >= n:
        return data
    else:
        step = n // sample_size
        sampled_data = [data[i] for i in range(0, n, step)]
        return sampled_data


def rotate_key_word_check(data_list):
    keys_count = 0

    keys = ["passport", "date", "country", "p ", "jpn ", "japan"]

    for data in data_list:
        text = data.content

        for key in keys:
            if key in text.lower():
                keys_count += 1

    if keys_count / len(keys) > 0.3:
        return True


def rotated_image_by_angle(image, angle):
    """
    根据旋转角度旋转图片
    """

    # 获取图像尺寸
    height, width = image.shape[:2]

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


# 获取识别文字的方向，并旋转图片，只能识别90 180 270
def rotate_image_with_white_bg(image):
    # 获取识别文字的位置信息
    angle = []
    data_list = ocr_by_key(image, "line", "jpn")

    if rotate_key_word_check(data_list):
        return image

    for i in range(3):
        angle = 90 * (i + 1)

        rotated_image = rotated_image_by_angle(image, angle)

        data_list = ocr_by_key(rotated_image, "line", "jpn")

        if rotate_key_word_check(data_list):
            return rotated_image

    raise ValueError("没有正确识别护照方向")


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


def get_overlap_percentage(normal_rect, ocr_data_rect):
    """
    判断orc数据是否在标准矩形区域内

    参数:
    - normal_rect: 一个包含四个元素的元组或列表，表示点的坐标 (x1, y1, x2, y2)
    - ocr_data_rect: 一个包含四个元素的元组或列表，表示矩形的坐标 (x1, y1, x2, y2)

    返回值:
    - 如果点在矩形区域内，则返回 True，否则返回 False
    """
    normal_area = (normal_rect[1][0] - normal_rect[0][0]) * (
        normal_rect[1][1] - normal_rect[0][1]
    )  # 计算正常矩形的面积

    intersection_x = max(
        0,
        min(normal_rect[1][0], ocr_data_rect[1][0])
        - max(normal_rect[0][0], ocr_data_rect[0][0]),
    )  # 计算交集的宽度
    intersection_y = max(
        0,
        min(normal_rect[1][1], ocr_data_rect[1][1])
        - max(normal_rect[0][1], ocr_data_rect[0][1]),
    )  # 计算交集的高度
    intersection_area = intersection_x * intersection_y  # 计算交集的面积

    overlap_percentage = (intersection_area / normal_area) * 100  # 计算重叠的百分比

    if overlap_percentage > 0.1:
        return overlap_percentage
    else:
        return 0


def check_len(ret):
    for title, key_len in Passport.PASSPORT_KEYS_LEN.items():
        if key_len > 0 and len(ret[title]) != key_len:
            ret["vs_info"][
                title
            ] = f"{Passport.OUT_ERROR_TAG}: 实际长度{len(ret[title])}不等于预测长度{key_len}"
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
    else:
        main_info[tmp] = ""

    main_info[Passport.Surname] = to_O(ret[Passport.Surname])
    main_info[Passport.Given_name] = to_O(ret[Passport.Given_name])
    main_info[Passport.Nationality] = to_O(ret[Passport.Nationality])

    tmp = Passport.Date_of_birth
    if vs_info[tmp][:5] != Passport.OUT_ERROR_TAG:
        main_info[tmp] = to_0(ret[tmp][:2]) + to_O(ret[tmp][2:5]) + to_0(ret[tmp][-4:])
    else:
        main_info[tmp] = ""

    main_info[Passport.Sex] = ret[Passport.Sex]

    main_info[Passport.Registered_Domicile] = to_O(ret[Passport.Registered_Domicile])

    tmp = Passport.Date_of_issue
    if vs_info[tmp][:5] != Passport.OUT_ERROR_TAG:
        main_info[tmp] = to_0(ret[tmp][:2]) + to_O(ret[tmp][2:5]) + to_0(ret[tmp][-4:])
    else:
        main_info[tmp] = ""

    tmp = Passport.Date_of_expiry
    if vs_info[tmp][:5] != Passport.OUT_ERROR_TAG:
        main_info[tmp] = to_0(ret[tmp][:2]) + to_O(ret[tmp][2:5]) + to_0(ret[tmp][-4:])
    else:
        main_info[tmp] = ""


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
        if title in main_info == False:
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


def rect_vs_box(rect, width, height):
    """
    矩形(x1,y1,x2,y2)数据不能超过设置范围
    """

    ((x1, y1), (x2, y2)) = rect

    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    x2 = width if x2 > width else x2
    y2 = height if y2 > height else y2

    return (x1, y1, x2, y2)


# 获取护照信息
def datalist2info(passport: Passport, data_list, img):
    ret = passport.info
    ret["main_info"] = {}
    ret["mrz_info"] = {}
    ret["vs_info"] = {}

    height, width = img.shape[:2]

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

    # 通过PassportNo确认位移
    passportNo_data = None
    for data in data_list:
        rect = data.position
        text = data.content

        pattern = r"^[a-zA-Z]{0,3}\d{5,9}$"
        match = re.match(pattern, text.strip().lower())
        if match:
            passportNo_data = data
            break

    x_offset = 0
    y_offset = 0
    if passportNo_data:
        ((x1, y1), (x2, y2)) = passportNo_data.position
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        ((px1, py1), (px2, py2)) = Passport.PASSPORT_KEYS_POSITION[Passport.Passport_No]
        p_center_x = (px1 + px2) / 2
        p_center_y = (py1 + py2) / 2

        # 计算中心点之间的偏移量
        x_offset = int(center_x - p_center_x)
        y_offset = int(center_y - p_center_y)

    for data in data_list:
        text = data.content
        ((x1, y1), (x2, y2)) = data.position

        if passportNo_data:
            # 移动矩形的坐标
            x1 = x1 - x_offset
            y1 = y1 - y_offset
            x2 = x2 - x_offset
            y2 = y2 - y_offset

        if len(text) <= 3:
            x1 -= 20
            x2 += 20

        y1 -= 10
        y2 += 10

        x1, y1, x2, y2 = rect_vs_box(((x1, y1), (x2, y2)), width, height)

        data.position = ((x1, y1), (x2, y2))

    # MRZ
    data_list_foots = [data for data in data_list if len(data.content) > 30]
    if len(data_list_foots) == 2:
        if data_list_foots[0].position[0][1] < data_list_foots[1].position[0][1]:
            ret[Passport.foot1] = data_list_foots[0].content
            ret[Passport.foot2] = data_list_foots[1].content
        else:
            ret[Passport.foot1] = data_list_foots[1].content
            ret[Passport.foot2] = data_list_foots[0].content

    else:
        for data in data_list_foots:
            text = data.content

            pattern = r"^P.JPN.*"
            match = re.match(pattern, text.strip().lower())
            if match:
                ret[Passport.foot1] = data.content

            pattern = r"^[a-zA-Z]{0,3}\d{5,9}$"
            match = re.match(pattern, text.strip().lower())
            if match:
                ret[Passport.foot2] = data.content

        if Passport.foot1 not in ret or not ret[Passport.foot1]:
            ret[Passport.foot1] = Passport.OUT_ERROR_TAG + ": 没有找到数据."
            error_vals += 1

        if Passport.foot2 not in ret or not ret[Passport.foot2]:
            ret[Passport.foot2] = Passport.OUT_ERROR_TAG + ": 没有找到数据."
            error_vals += 1

    for key, value in Passport.PASSPORT_KEYS_POSITION.items():
        if "foot" in key:
            continue

        # 统计所有覆盖率
        overlap_percentages = []
        for data in data_list:
            rect = data.position
            text = data.content

            overlap_percentage = get_overlap_percentage(value, rect)
            if overlap_percentage != 0:
                overlap_percentages.append((overlap_percentage, text))

        # 找到覆盖率最大的值
        if len(overlap_percentages) != 0:
            max_overlap_percentage = 0
            max_overlap_text = ""
            for overlap_percentage, text in overlap_percentages:
                if overlap_percentage > max_overlap_percentage:
                    max_overlap_percentage = overlap_percentage
                    max_overlap_text = text
            ret[key] = max_overlap_text

        if key not in ret:
            ret[key] = Passport.OUT_ERROR_TAG + ": 没有找到数据."
            error_vals += 1

    if error_vals > 0:
        err_msg = f"一共有{error_vals}个数据没有找到对应值。"
        ret["err_msg"] = add_error_to_info(ret["err_msg"], err_msg)

    # 如果国籍和年月日黏在一起
    if ret[Passport.Nationality] == ret[Passport.Date_of_birth]:
        if len(ret[Passport.Nationality]) > 8:
            ret[Passport.Date_of_birth] = ret[Passport.Date_of_birth][-9:]
            ret[Passport.Nationality] = ret[Passport.Nationality][:-9]
        else:
            ret[
                Passport.Nationality
            ] = f"{Passport.OUT_ERROR_TAG} : 数据错误. {ret[Passport.Nationality]}"
            ret[Passport.Date_of_birth] = ret[Passport.Nationality]

    # 根据基础信息生成三个对象，对象main_info保存护照主要信息，对象mrz_info保存下方mrz分解后的信息，对象vs_info保存对比信息
    check_len(ret)
    set_main_info(ret)
    set_mrz_info(ret)
    set_vs_info(ret)

    print(f"err_msg:{ret['err_msg']}")

    return ret, data_list


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


def get_rotate(image):
    # 缩放到固定高度800
    image = resize_image_by_height(image, 1000)

    h, w = image.shape[:2]
    y1 = int(h * 0.2)
    y2 = int(h * 0.8)
    image = image[y1:y2, 0:w]

    # 增加对比度
    _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    # 使用Canny边缘检测算法检测边缘
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # 使用霍夫直线变换检测直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    temp_lines = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            # 过滤水平方向的直线
            if abs(theta) < np.pi / 180 * 80 or abs(theta - np.pi) < np.pi / 180 * 80:
                continue

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            temp_lines.append(((x1, y1, x2, y2), (rho, theta)))

    return temp_lines, binary


def find_passport_line(data_list):
    """
    从OCR结果获取passport所在行
    """
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

    return passport_data


def rotate_rectangle(x1, y1, x2, y2, angle):
    # 计算矩形的中心点坐标
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # 创建旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    # 定义矩形的四个顶点坐标
    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    # 使用旋转矩阵对矩形的四个顶点坐标进行旋转变换
    rotated_points = cv2.transform(np.array([points]), rotation_matrix)[0]

    # 计算旋转后的矩形的最小外接矩形
    rotated_x, rotated_y, rotated_w, rotated_h = cv2.boundingRect(rotated_points)

    # 计算旋转后矩形的四个顶点坐标
    x1_rot = rotated_x
    y1_rot = rotated_y
    x2_rot = rotated_x + rotated_w
    y2_rot = rotated_y + rotated_h

    return int(x1_rot), int(y1_rot), int(x2_rot), int(y2_rot)


def find_passport_area(img):
    data_list_line = ocr_by_key(img, "line", "jpn")
    data_list = [data for data in data_list_line if len(data.content) > 5]

    if debug_mode:
        img_test = rect_set(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), data_list)
        _imshow("test", img_test)

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
    passport_data = find_passport_line(data_list)

    # data_list_line = ocr_by_key(img, "line", "jpn")

    mrz_1 = mrz_datas[0].position
    mrz_2 = mrz_datas[1].position

    mrz_t = mrz_1
    if mrz_t[0][0] < mrz_2[0][0]:
        mrz_t = mrz_2

    # 旋转图片，还原倾斜度 y1:y2 x1:x2
    if passport_data:
        y1 = passport_data.position[0][1] - 15 - 1200
    else:
        y1 = mrz_2[1][1] - (mrz_2[1][1] - mrz_2[0][1]) * 25 - 200

    y1 = 0 if y1 < 0 else y1

    cut_img = img[
        y1 : mrz_2[1][1],
        mrz_t[0][0] - 5 : mrz_t[1][0] + 5,
    ]

    # 旋转图片，还原倾斜度 y1:y2 x1:x2
    lines, rotate_img = get_rotate(img)

    angle = 0
    rotate_img = img.copy()
    if len(lines) > 0:
        rho, theta = lines[0][1]
        angle = -np.degrees(np.pi / 2 - theta)

        if debug_mode:
            print(f"angle: {angle}")

        if abs(angle) > 0.5:
            rotate_img = rotated_image_by_angle(rotate_img, angle)

    # 获得图片
    if passport_data:
        y1 = passport_data.position[0][1] - 115
    else:
        y1 = mrz_2[1][1] - (mrz_2[1][1] - mrz_2[0][1]) * 25

    x1 = mrz_t[0][0] - 5
    x2 = mrz_t[1][0] + 5
    y1 = y1
    y2 = mrz_2[1][1]
    height, width = img.shape[:2]
    if abs(angle) > 0.5:
        # print(x1, y1, x2, y2,height, width)
        x1, y1, x2, y2 = rotate_rectangle(x1, y1, x2, y2, angle)

        if angle > 0:
            x1 += 80
            x2 += 80
            y1 += 5
            y2 += 5
        else:
            x1 += 10
            x2 -= 10
            y1 += 5
            y2 += 5

    height, width = rotate_img.shape[:2]
    # print(x1, y1, x2, y2,height, width)
    x1, y1, x2, y2 = rect_vs_box(((x1, y1), (x2, y2)), width, height)

    cut_img = rotate_img[y1:y2, x1:x2]

    return cut_img

# 识别后数据输出到文本文件中
def output_data2text_file(passport, _config_options: dict):

    output_data_file = (
        _config_options["OUTPUT_FOLDER_PATH"]
        + "/"
        + Passport.json_dir
        + "/"
        + passport.file_name
        + ".json"
    )
    # 打开文件，将文件指针移动到文件的末尾
    with open(output_data_file, "a", encoding="utf-8") as f:
        json.dump(passport.info, f, ensure_ascii=False)

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
    gray_image = rotate_image_with_white_bg(thresh)

    # plt.imsave("to_rotate" + "\\" + passport.file_name + "_rotate.png", cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

    # pil_image = Image.open("to_rotate" + "\\" + passport.file_name + "_rotate.png")
    # gray_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)

    # 截取护照
    gray_image = find_passport_area(gray_image)

    # 二值化图像
    _, binary = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

    gray_image_OCR = fill_middle_with_white(binary.copy(), "下边涂白")

    data_list_word = ocr_by_key(gray_image_OCR, "word", "jpn")

    data_list_line = ocr_by_key(binary, "line", "jpn")

    cut_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    img_cv = rect_set(cut_img, data_list_word)

    plt.imsave(sample_cut_img_path, img_cv)

    # 获取数据范围
    thresh = get_mask_rect(data_list_word, data_list_line, gray_image)
    # 缩放到固定高度800
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
    img = add_border_to_grayscale_image(img, 100)

    # OCR
    data_list = ocr_by_key(img, "word", "num_1")

    dict_array = []
    top_data = None
    for data in data_list:
        rect = data.position
        text = data.content

        if top_data:
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

    y1 = top_data.position[0][1] - 52
    x1 = max(mrz1.position[0][0], mrz2.position[0][0]) - 44

    class Ocr_ret:
        def __init__(self, position, content):
            self.position = position
            self.content = content

    _data_list = []
    for data in data_list:
        rect = data.position
        text = data.content

        _rect = ((rect[0][0] - x1, rect[0][1] - y1), (rect[1][0] - x1, rect[1][1] - y1))

        _data_list.append(Ocr_ret(_rect, text))

    height, width = img.shape[:2]

    img = img[y1:height, x1:width]

    cut_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 获取护照信息
    passport.info, data_list = datalist2info(passport, _data_list, cut_img)

    # 存储OCR结果图片
    img_cv = rect_set(cut_img, _data_list)

    for key, i_rect in Passport.PASSPORT_KEYS_POSITION.items():
        x1, y1, x2, y2 = i_rect[0][0], i_rect[0][1], i_rect[1][0], i_rect[1][1]
        cv2.rectangle(img_cv, i_rect[0], i_rect[1], (255, 0, 0), 2)

    # img_cv = cut_img
    plt.imsave(sample_edited_img_path, img_cv)

    # 识别后数据输出到文本文件中
    output_data2text_file(passport, config_options)


def run(passport: Passport, _config_options: dict):
    try:
        main(passport, _config_options)
    except Exception as e:
        # 捕获异常并打印错误信息
        print(f"发生错误 {passport.file_name}:", str(e))

        ret = passport.info
        ret["err_msg"] = add_error_to_info(ret["err_msg"], str(e))