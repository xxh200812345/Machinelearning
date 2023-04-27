import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyocr
import pyocr.builders
import cv2
import string
import platform

# 目标图
sample_img_path = "sample.jpg"
sample_edited_img_path = "sample_edited.png"
sample_sign_img_path = "sample_sign.png"

# 获取操作系统名称及版本号
os = platform.system()
pyocr.tesseract.TESSERACT_CMD = "E:/Program Files/Tesseract-OCR/tesseract.exe"

# 判断当前操作系统
if os == "Darwin":
    pyocr.tesseract.TESSERACT_CMD = "/opt/homebrew/Cellar/tesseract/5.3.1/bin/tesseract"


def get_original_img(path, mode):
    img = cv2.imread(path, mode)
    # 调整图像大小以便于处理
    img = cv2.resize(img, (600, 420))
    return img


def ocr_by_key(img, key):
    # ツール取得
    tool = pyocr.get_available_tools()[0]

    # OCR
    builder = None
    digits = None

    if key == "line":
        builder = pyocr.builders.LineBoxBuilder(tesseract_layout=6)

    elif key == "digits":
        # chars = string.ascii_uppercase + string.digits + "<"
        # digits_config = (
        #     f"--psm 6 tessedit_char_whitelist {chars}"  # Only recognize digits
        # )
        builder = pyocr.builders.DigitLineBoxBuilder(tesseract_layout=6)

    elif key == "word":
        builder = pyocr.builders.WordBoxBuilder(tesseract_layout=6)

    else:
        print("key is error.")

    digits = tool.image_to_string(Image.fromarray(img), lang="num_1", builder=builder)

    return digits


# 異なる背景除去
def render_doc_text(file_path):
    # グレースケールでイメージを読み込み
    img_generated = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    print("img shape", img_generated.shape)

    # 異なる背景除去
    ret, img_adaptive = cv2.threshold(img_generated, 200, 255, cv2.THRESH_BINARY)

    return img_adaptive


# 标记识别结果，并显示图片
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
        font_path = "NotoSansJP-Thin.otf"
        font_size = 16
        font_color = (0, 0, 255)

        # 加载日语字体
        font = ImageFont.truetype(font_path, font_size)

        # 在图像上绘制文本
        draw.text((x1, y1 - 23), text, font=font, fill=font_color)

    # 将图像从Pillow格式转换为OpenCV格式
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return img_cv


# 获取前两位最长的文字列的位置，返回他们的矩阵
def get_foot_area():
    img = cv2.imread(sample_edited_img_path, cv2.IMREAD_GRAYSCALE)

    # OCR検知
    data_list = ocr_by_key(img, "line")

    text_arr = np.array([data.content for data in data_list])

    # 求出每个文字列的长度
    text_lens = np.array([len(text) for text in text_arr])

    # 找出最长的3个文字列
    longest_idx = text_lens.argsort()[-2:][::-1]

    x_array = [
        data_list[longest_idx[0]].position[0][0],
        data_list[longest_idx[0]].position[1][0],
        data_list[longest_idx[1]].position[0][0],
        data_list[longest_idx[1]].position[1][0],
    ]
    y_array = [
        data_list[longest_idx[0]].position[0][1],
        data_list[longest_idx[0]].position[1][1],
        data_list[longest_idx[1]].position[0][1],
        data_list[longest_idx[1]].position[1][1],
    ]

    x1 = min(x_array)
    x2 = max(x_array)
    y1 = min(y_array)
    y2 = max(y_array)

    return (
        (
            x1 - 2,
            y1 - 2,
        ),
        (
            x2 + 2,
            y2 + 2,
        ),
    )


# 获取右侧区域的右上角坐标
def get_right_area():
    x, y = -1, -1

    img = cv2.imread(sample_edited_img_path, cv2.IMREAD_GRAYSCALE)

    # OCR検知
    data_list = ocr_by_key(img, "word")
    find_passport = False
    for data in data_list:
        rect = data.position
        # 定义要绘制的文本和位置
        text = data.content

        if find_passport:
            x = rect[0][0] - 2
            y = rect[0][1] - 2
            break

        if "PASSPORT" == text:
            find_passport = True

    return (x, y-8)


def mask_fill_white(img, mask):
    # 将矩形取反
    mask_inv = cv2.bitwise_not(mask)

    # 将图像和矩形取出矩形内部区域
    img_bg = cv2.bitwise_and(img, img, mask=mask)

    # 将图像和矩形取出矩形外部区域
    img_fg = cv2.bitwise_and(img, img, mask=mask_inv)

    # 创建一个白色背景
    background = np.ones(img.shape, dtype=np.uint8) * 255

    # 将矩形外部区域涂成白色
    result = cv2.add(img_fg, background, mask=mask_inv)

    # 将矩形内部区域与矩形外部区域合并
    result = cv2.add(img_bg, result)

    return result


def _imshow(title, img):
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
    #_imshow("Gaussian Thresholding", bin_clo)

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


# 图片二值化，把白色部分设置为透明
def binary_img_with_transparency(img, threshold=180):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = color_scale_display(gray, 112, 217, 0.97)

    # 将二值化后的图像转换为4通道图像
    rgba = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGBA)

    # 将白色部分设置为透明
    rgba[:, :, 3] = np.where(thresh == 255, 0, 255)
    return rgba


# 裁切不需要的部分,返回处理后的
def get_main_area():
    # 異なる背景除去
    img = render_doc_text(sample_img_path)
    cv2.imwrite(sample_edited_img_path, img)

    # 获取前两位最长的文字列的位置，返回他们的矩阵 ((x1,y1),(x2,y2))
    foot_area = get_foot_area()
    print(f"foot_area: {foot_area}")
    # cv2.rectangle(img, foot_area[0], foot_area[1], (0, 0, 255), 2)
    # _imshow("test", img)
    # exit()

    # 获取右侧区域的右上角坐标 (x,y)
    right_point = get_right_area()
    print(f"right_point: {right_point}")

    # 获取签名切片操作
    img = get_original_img(sample_img_path, cv2.IMREAD_COLOR)
    # ((y1,y2),(x1,x2)) 识别到的签名范围
    sign_rect = (
        (
            right_point[1] + int((foot_area[0][1] - right_point[1]) * 0.6),
            foot_area[0][1],
        ),
        (
            (right_point[0] + int((foot_area[1][0] - right_point[0]) * 0.4)),
            foot_area[1][0],
        ),
    )
    img_part = img[sign_rect[0][0] : sign_rect[0][1], sign_rect[1][0] : sign_rect[1][1]]
    # 图片二值化，把白色部分设置为透明
    img_part = binary_img_with_transparency(img_part)
    cv2.imwrite(sample_sign_img_path, img_part)

    # グレースケールでイメージを読み込み
    img = get_original_img(sample_img_path, cv2.IMREAD_GRAYSCALE)
    img = color_scale_display(img, 112, 217, 0.97)

    # 读取原始图像和矩形遮罩
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # 绘制底部矩形遮罩
    cv2.rectangle(mask, foot_area[0], foot_area[1], 255, -1)
    # 绘制右边矩形遮罩
    cv2.rectangle(mask, right_point, (foot_area[1][0], foot_area[0][1]), 255, -1)

    # 遮罩外涂白
    img = mask_fill_white(img, mask)

    # 只返回指定高度以内的区域（max，min）
    img_mask = get_original_img(sample_img_path, cv2.IMREAD_GRAYSCALE)
    # 二值化图像
    _, img_mask = cv2.threshold(img_mask, 150, 255, cv2.THRESH_BINARY)
    mask = remove_small_height_regions(img_mask, 15, 10)

    # 遮罩外涂白
    img = mask_fill_white(img, mask)

    # 签名区域内涂白
    cv2.rectangle(
        img,
        (sign_rect[1][0], sign_rect[0][0]),
        (sign_rect[1][1], sign_rect[0][1]),
        255,
        -1,
    )
    cv2.imwrite(sample_edited_img_path, img)


if __name__ == "__main__":
    # 裁切不需要的部分,返回处理后的
    get_main_area()

    # 读取图像处理后的图片
    img = cv2.imread(sample_edited_img_path, cv2.IMREAD_GRAYSCALE)

    # OCR
    data_list = ocr_by_key(img, "digits")

    # 标记识别结果，并显示图片
    img = cv2.imread(sample_edited_img_path, cv2.IMREAD_COLOR)

    img_cv = rect_set(img, data_list)

    img_sign = cv2.imread(sample_sign_img_path, cv2.IMREAD_COLOR)
    _imshow("img_sign", img_sign)

    img_sample = cv2.imread(sample_img_path, cv2.IMREAD_COLOR)

    # 获取图像大小
    h, w, c = img_cv.shape

    # 调整图像大小
    img_sample = cv2.resize(img_sample, (w, h))

    # 水平拼接图像
    h_concat = np.concatenate((img_sample, img_cv), axis=1)

    # 显示拼接后的图像
    _imshow("Horizontal Concatenation", h_concat)