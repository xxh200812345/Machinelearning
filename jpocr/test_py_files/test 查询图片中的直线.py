import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import pyocr
import re


# 设置tessract入口程序安装位置
def set_tessract_app():
    pyocr.tesseract.TESSERACT_CMD = r"E:\Program Files\Tesseract-OCR\tesseract.exe"


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


def find_nearest_lines(lines, image_height):
    center_y = image_height // 2
    upper_lines = []
    lower_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        mid_y = (y1 + y2) // 2

        if mid_y < center_y:
            upper_lines.append(line)
        else:
            lower_lines.append(line)

    upper_nearest_line = find_nearest_line(upper_lines, center_y)
    lower_nearest_line = find_nearest_line(lower_lines, image_height - center_y)

    return upper_nearest_line, lower_nearest_line


def find_nearest_line(lines, target_y):
    nearest_line = None
    min_distance = float("inf")

    for line in lines:
        x1, y1, x2, y2 = line[0]
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2

        distance = abs(mid_y - target_y)

        if distance < min_distance:
            min_distance = distance
            nearest_line = line

    return nearest_line


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


def get_rotate(image):
    # 缩放到固定高度800
    image = resize_image_by_height(image, 1000)

    h, w = image.shape[:2]
    y1 = int(h * 0.2)
    y2 = int(h * 0.8)
    image = image[y1:y2, 0:w]

    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 增加对比度
    _, img_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 使用Canny边缘检测算法检测边缘
    edges = cv2.Canny(img_mask, 50, 150, apertureSize=3)

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

    image = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)

    # 绘制检测到的直线
    for x1, y1, x2, y2 in [line[0] for line in temp_lines]:
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return temp_lines, image


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


def find_passport_area(img):
    data_list_line = ocr_by_key(img, "line", "jpn")
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

    # data_list_line = ocr_by_key(img, "line", "jpn")

    mrz_1 = mrz_datas[0].position
    mrz_2 = mrz_datas[1].position

    mrz_t = mrz_1
    if mrz_t[0][0] < mrz_2[0][0]:
        mrz_t = mrz_2

    # 获得图片
    if passport_data:
        y1 = passport_data.position[0][1] - 15 - 1200
    else:
        y1 = mrz_2[1][1] - (mrz_2[1][1] - mrz_2[0][1]) * 25 - 200

    y1 = 0 if y1 < 0 else y1

    img = img[
        y1 : mrz_2[1][1],
        mrz_t[0][0] - 5 : mrz_t[1][0] + 5,
    ]

    # 旋转图片，还原倾斜度 y1:y2 x1:x2
    mrz_1_img = img[
        mrz_1[0][1] : mrz_1[1][1],
        mrz_1[0][0] : mrz_1[1][0],
    ]

    mrz_1_img

    return img


def line_segment_detection(image, threshold=50):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 直线检测
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold, minLineLength=50, maxLineGap=10
    )

    # 创建输出图像
    output = np.zeros_like(image)

    # 绘制检测到的直线段
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return output


if __name__ == "__main__":
    set_tessract_app()

    folder_path = r"H:\vswork\Machinelearning\jpocr\to_rotate"
    file_names = os.listdir(f"{folder_path}/1")
    for file_name in file_names:
        file_name = os.path.basename(file_name)
        extension = os.path.splitext(file_name)[1]

        if "png" not in extension:
            continue

        # 使用PIL库打开图像文件
        path = f"{folder_path}/1/{file_name}"
        pil_image = Image.open(path)
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # img = find_passport_area(img)

        # plt.imsave(f"{folder_path}/1/{file_name}", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # continue

        lines, img = get_rotate(img)

        if len(lines) > 0:
            rho, theta = lines[0][1]
            angle = -np.degrees(np.pi/2 - theta)

            print(f"{path} angle: {angle}")
            img = rotated_image_by_angle(img, angle)


        plt.imsave(
            f"{folder_path}\output\{file_name}", cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )
