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
    res = add_border_to_grayscale_image(res,border)

    return res

#白底灰度图像边框加宽
def add_border_to_grayscale_image(image, border_size=10, border_color=255):
    # 获取图像的尺寸
    image_height, image_width = image.shape

    # 计算背景的尺寸
    background_height = image_height + (2 * border_size)
    background_width = image_width + (2 * border_size)

    # 创建背景图像
    background = np.full((background_height, background_width), border_color, dtype=np.uint8)

    # 将图像放置在背景中心
    x = (background_width - image_width) // 2
    y = (background_height - image_height) // 2
    background[y:y+image_height, x:x+image_width] = image

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


def _imshow(title, img):
    # if debug_mode == True:
    cv2.imshow(title, img)
    cv2.waitKey(0)  # 等待用户按下键盘上的任意键
    cv2.destroyAllWindows()  # 关闭所有cv2.imshow窗口


# 只返回指定高度以内的区域（max，min）
def remove_small_height_regions(img, max_height, min_height):
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

    # 读取原始图像和矩形遮罩
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
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
    mask = remove_small_height_regions(img_mask, 20, 3)

    # 遮罩外涂白
    img = mask_fill_white(thresh, mask)

    data_list = ocr_by_key(img, "word", "num_1")
    cut_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_cv = rect_set(cut_img, data_list)
    cv2.imwrite(sample_edited_img_path, img_cv)


# 参数初始化
# 删除空白区域
# 清楚无用信息
