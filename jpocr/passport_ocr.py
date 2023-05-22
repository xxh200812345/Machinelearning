import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyocr
import pyocr.builders
import cv2
import platform
from collections import Counter

from PIL import Image

import json
from passport import Passport
from datetime import datetime

import pytesseract


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
    global sample_img_path, sample_cut_img_path, sample_edited_img_path, sample_sign_img_path
    global debug_mode, debug_font

    input_dir = config_options["PASSPORT_IMAGES_FOLDER_PATH"]
    output_dir = config_options["OUTPUT_FOLDER_PATH"]

    if passport.ext == ".pdf":
        sample_img_path = f"{output_dir}/{passport.pdf2png_file_name}"
    else:
        sample_img_path = f"{input_dir}/{passport.file_name}"
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


def get_keywords(data_list, img):
    passport_rect = None
    # 找到 PASSPORT
    for data in data_list:
        rect = data.position
        text = data.content
        if text.strip() == "PASSPORT":
            passport_rect = rect
            print(data)
            break

    if passport_rect == None:
        raise ValueError("PASSPORT关键字识别失败")

    # 最大的字符串
    max_data = None
    # 和PASSPORT最接近的字符串
    near_datas = []
    # PASSPORT的中轴线
    passport_mid_h = (passport_rect[0][1] + passport_rect[1][1]) / 2
    # PASSPORT的高
    passport_h = passport_rect[1][1] - passport_rect[0][1]
    for data in data_list:
        rect = data.position
        text = data.content
        # 中轴线
        data_mid_h = (rect[0][1] + rect[1][1]) / 2
        # 高
        data_h = rect[1][1] - rect[0][1]

        if (
            abs(data_mid_h - passport_mid_h) < passport_h
            and passport_h < int(passport_h * 0.2) + data_h
            and passport_rect[1][0] < rect[1][0]
        ):
            near_datas.append(data)

        if max_data == None or len(text) > len(max_data.content):
            max_data = data

    print(f"max_data: {max_data}")

    p_rect = None
    passportno_rect = None
    for data in near_datas:
        rect = data.position
        text = data.content

        # if debug_mode:
        print(f"{text}: {rect}")

        if "p" in text.strip().lower() and len(text) < 3:
            p_rect = data.position
            continue

        if abs(len(text) - 9) < 3:
            passportno_rect = data.position

    if p_rect == None:
        raise ValueError("P关键字识别失败")
    if passportno_rect == None:
        raise ValueError("Passport No关键字识别失败")

    ret = {"passport": passport_rect, "max_data": max_data}

    # 获取图片的宽度和高度
    height, width = img.shape[:2]

    # 创建一个全白色的遮罩，它的大小和原图一样
    mask = np.ones((height, width), dtype="uint8") * 255

    # 定义多边形顶点坐标
    max_data_rect = max_data.position
    max_data_h = max_data_rect[1][1] - max_data_rect[0][1]
    points = np.array(
        [
            [p_rect[0][0] - 2, passportno_rect[0][1] - 2],  # 左上
            [passportno_rect[1][0] + 2, passportno_rect[0][1] - 2],
            [max_data_rect[1][0] + 2, passportno_rect[0][1] - 2],  # 右上
            [max_data_rect[1][0] + 2, max_data_rect[1][1] + 2],
            [max_data_rect[0][0] - 2, max_data_rect[1][1] + 2],
            [max_data_rect[0][0] - 2, max_data_rect[1][1] - max_data_h * 3],
            [p_rect[0][0] - 2, max_data_rect[1][1] - max_data_h * 3],
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
    res = res[
        passportno_rect[0][1] - border : max_data_rect[1][1] + border,
        max_data_rect[0][0] - border : max_data_rect[1][0] + border,
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


# 获取识别文字的方向，并旋转图片，只能识别90 180 270
def rotate_image_with_white_bg(image):
    # 获取识别文字的位置信息
    data = pytesseract.image_to_osd(image)
    print(data)
    # 从位置信息中提取文字方向
    lines = data.split("\n")
    angle = None
    for line in lines:
        if line.startswith("Orientation in degrees:"):
            angle = float(line.split(":")[1].strip())
            break

    if angle < 1:
        return image

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


def _imshow(title, img):
    # if debug_mode == True:
    cv2.imshow(title, img)
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
    output_data_file = _config_options["OUTPUT_FOLDER_PATH"] + "/data.json"

    # 打开文件，将文件指针移动到文件的末尾
    with open(output_data_file, "a", encoding="utf-8") as f:
        json.dump([passport.info for passport in passport_list], f, ensure_ascii=False)


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
                    Surname_p2 = position[0]+len(mrz_info[title])
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


def add_error_to_vsinfo(info, error_msg):
    """
    追加对应的比较错误 info：vs_info[title]
    """
    if info[:5] != Passport.OUT_ERROR_TAG:
        info = Passport.OUT_ERROR_TAG + ": " + error_msg
    else:
        info += ";" + error_msg

    return info


def set_vs_info(ret):
    main_info = ret["main_info"]
    mrz_info = ret["mrz_info"]
    vs_info = ret["vs_info"]

    for title, mrz_item in mrz_info.items():
        if main_info[title][:5] == Passport.OUT_ERROR_TAG:
            error_msg = f"中间的信息项目存在错误"
            vs_info[title] = add_error_to_vsinfo(vs_info[title], error_msg)

        elif mrz_info[title][:5] == Passport.OUT_ERROR_TAG:
            error_msg = f"MRZ的信息项目存在错误"
            vs_info[title] = add_error_to_vsinfo(vs_info[title], error_msg)

        elif main_info == "":
            error_msg = f"中间的信息项目没有值"
            vs_info[title] = add_error_to_vsinfo(vs_info[title], error_msg)

        elif mrz_info == "":
            error_msg = f"MRZ的信息项目没有值"
            vs_info[title] = add_error_to_vsinfo(vs_info[title], error_msg)

        elif title == Passport.Date_of_birth or title == Passport.Date_of_expiry:
            month_num = get_month_number(main_info[title][2:5])
            if isinstance(month_num, str):
                error_msg = month_num
                vs_info[title] = add_error_to_vsinfo(vs_info[title], error_msg)
            else:
                main_item = (
                    main_info[title][-2:] + str(month_num).zfill(2) + main_info[title][:2]
                )
                if main_item != mrz_item:
                    error_msg = f"数据不一致"
                    vs_info[title] = add_error_to_vsinfo(vs_info[title], error_msg)

        elif title == Passport.Nationality:
            nationality_code=Passport.get_nationality_code(main_info[title])
            if nationality_code == 'UNK':
                error_msg = f"没找到对应的国家（{main_info[title]}）code"
                vs_info[title] = add_error_to_vsinfo(vs_info[title], error_msg)

            elif nationality_code != mrz_info[title]:
                error_msg = f"数据不一致"
                vs_info[title] = add_error_to_vsinfo(vs_info[title], error_msg)

        elif main_info[title] != mrz_info[title]:
            error_msg = f"数据不一致"
            vs_info[title] = add_error_to_vsinfo(vs_info[title], error_msg)

    return


# 获取护照信息
def datalist2info(passport: Passport, data_list):
    ret = {}
    ret["main_info"] = {}
    ret["mrz_info"] = {}
    ret["vs_info"] = {}
    ret["err_msg"] = ""

    ocr_texts = ""
    for data in data_list:
        # 文本
        ocr_texts += f"{data.content} {data.position}\n"

    ret["file_name"] = passport.file_name

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ret["time"] = now_str

    if len(data_list) != 13:
        ret["err_msg"] += "识别后文字信息行数不为13，有识别错误。"

    ret["ocr_texts"] = ocr_texts

    # for i, data in enumerate(data_list):
    #     rect = data.position
    #     text = data.content
    #     ((x1, y1), (x2, y2)) = rect
    #     rect_center = (int(x1 + x2) / 2, int(y1 + y2) / 2)
    #     print(f"'{PASSPORT_TITLES[i]}' , {rect_center}")

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
        ret["err_msg"] += f"一共有{error_vals}个数据没有找到对应值。"

    # 根据基础信息生成三个对象，对象main_info保存护照主要信息，对象mrz_info保存下方mrz分解后的信息，对象vs_info保存对比信息
    check_len(ret)
    set_main_info(ret)
    set_mrz_info(ret)
    set_vs_info(ret)

    return ret


def run(passport: Passport, _config_options: dict):
    global config_options

    config_options = _config_options

    # 初始化设置
    init(passport)

    # 裁切30px不需要的部分,返回处理后的
    img = cv2.imread(sample_img_path)
    del_img = crop_image(img, 30)
    # 获取识别文字的方向，并旋转图片，只能识别90 180 270
    del_img = rotate_image_with_white_bg(del_img)

    gray = cv2.cvtColor(del_img, cv2.COLOR_BGR2GRAY)
    thresh = color_scale_display(gray, 112, 217, 0.97)

    x1, y1, x2, y2 = calc_row_and_col_sum(thresh)

    # 使用切片操作截取图像的中心部分top:bottom, left:right
    thresh = thresh[int(y1 + (y2 - y1) / 2) : y2, x1:x2]
    data_list = ocr_by_key(thresh, "word", "jpn")
    cut_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    img_cv = rect_set(cut_img, data_list)
    cv2.imwrite(sample_cut_img_path, img_cv)

    # 获取数据范围
    thresh = get_keywords(data_list, thresh)
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
    cv2.imwrite(
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

    # OCR
    data_list = ocr_by_key(img, "word", "num_1")
    cut_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 获取护照信息
    passport.info = datalist2info(passport, data_list)

    # 存储OCR结果图片
    img_cv = rect_set(cut_img, data_list)
    cv2.imwrite(sample_edited_img_path, img_cv)
