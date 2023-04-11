from PIL import Image, ImageDraw, ImageFont
import pyocr
import pyocr.builders
import cv2
import numpy as np

# 目标图
sample_img_path = "sample.jpg"
sample_edited_img_path = "sample_edited.png"
sample_sign_img_path = "sample_sign.png"
pyocr.tesseract.TESSERACT_CMD = "E:/Program Files/Tesseract-OCR/tesseract.exe"

def get_original_img(path, mode):
    img = cv2.imread(path, mode)
    # 调整图像大小以便于处理
    img = cv2.resize(img, ( 600,420))
    return img


def ocr_by_key(key):
    # ツール取得
    tools = pyocr.get_available_tools()
    tool = tools[0]

    # グレースケールでイメージを読み込み
    img = cv2.imread(sample_edited_img_path, cv2.IMREAD_GRAYSCALE)

    # OCR
    builder = None
    if key == "line":
        builder = pyocr.builders.LineBoxBuilder(tesseract_layout=6)
    else:
        builder = pyocr.builders.WordBoxBuilder(tesseract_layout=6)

    return tool.image_to_string(Image.fromarray(img), lang="eng", builder=builder)


# 異なる背景除去
def render_doc_text(file_path):
    # グレースケールでイメージを読み込み
    img_generated = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    print("img shape", img_generated.shape)

    # 異なる背景除去
    ret, img_adaptive = cv2.threshold(img_generated, 200, 255, cv2.THRESH_BINARY)

    return img_adaptive


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
        font_path = "H:/vswork/AI/jpocr/NotoSansJP-Thin.otf"
        font_size = 16
        font_color = (0, 0, 255)

        # 加载日语字体
        font = ImageFont.truetype(font_path, font_size)

        # 在图像上绘制文本
        draw.text((x1, y1 - 23), text, font=font, fill=font_color)

    # 将图像从Pillow格式转换为OpenCV格式
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("img with rectangles", img_cv)
    cv2.waitKey(0)


# 获取前两位最长的文字列的位置，返回他们的矩阵
def get_foot_area():
    # OCR検知
    data_list = ocr_by_key("line")

    text_arr = np.array([data.content for data in data_list])

    # 求出每个文字列的长度
    text_lens = np.array([len(text) for text in text_arr])

    # 找出最长的3个文字列
    longest_idx = text_lens.argsort()[-2:][::-1]

    return (
        (
            data_list[longest_idx[0]].position[0][0] - 2,
            data_list[longest_idx[0]].position[0][1] - 2,
        ),
        (
            data_list[longest_idx[1]].position[1][0] + 2,
            data_list[longest_idx[1]].position[1][1] + 2,
        ),
    )


# 获取右侧区域的右上角坐标
def get_right_area():
    x, y = -1, -1

    # OCR検知
    data_list = ocr_by_key("line")

    for data in data_list:
        rect = data.position
        # 定义要绘制的文本和位置
        text = data.content

        if ("p" in text or "P" in text) and "PASSPORT" in text:
            y = rect[0][1] - 2
            break

    # OCR検知
    data_list = ocr_by_key("word")
    find_passport = False
    for data in data_list:
        rect = data.position
        # 定义要绘制的文本和位置
        text = data.content

        if find_passport:
            x = rect[0][0] - 2
            break

        if "PASSPORT" == text:
            find_passport = True

    return (x, y)


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


# 只返回指定高度以内的区域（max，min）
def remove_small_height_regions(max_height, min_height):
    img = cv2.imread(sample_edited_img_path, cv2.IMREAD_GRAYSCALE)
    # 二值化图像
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # 对输入图像取反
    inverted_img = cv2.bitwise_not(binary)
    # 膨胀操作
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    bin_clo = cv2.dilate(inverted_img, kernel2, iterations=2)
    cv2.imshow("bin_clo", bin_clo)
    cv2.waitKey(0)

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


# 图片二值化，把白色部分设置为透明
def binary_img_with_transparency(img, threshold=180):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow("Result", thresh)
    cv2.waitKey(0)

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
    # print(f"foot_area: {foot_area}")

    # 获取右侧区域的右上角坐标 (x,y)
    right_point = get_right_area()
    # print(f"right_point: {right_point}")

    # 获取签名切片操作
    img = get_original_img(sample_img_path, cv2.IMREAD_COLOR)
    # ((y1,y2),(x1,x2))
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
    ret, img = cv2.threshold(img, 150, 250, cv2.THRESH_BINARY)
    
    # 读取原始图像和矩形遮罩
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # 绘制底部矩形遮罩
    cv2.rectangle(mask, foot_area[0], foot_area[1], 255, -1)
    # 绘制右边矩形遮罩
    cv2.rectangle(mask, right_point, (foot_area[1][0], foot_area[0][1]), 255, -1)

    # 遮罩外涂白
    img = mask_fill_white(img, mask)
    cv2.imwrite(sample_edited_img_path, img)

    # 只返回指定高度以内的区域（max，min）
    mask = remove_small_height_regions(15, 10)

    # 遮罩外涂白
    img = cv2.imread(sample_edited_img_path, cv2.IMREAD_GRAYSCALE)
    img = mask_fill_white(img, mask)

    #签名区域内涂白
    cv2.rectangle(img, (sign_rect[1][0],sign_rect[0][0]),(sign_rect[1][1],sign_rect[0][1]), 255, -1)
    cv2.imwrite(sample_edited_img_path, img)

    data_list = ocr_by_key("word")

    return data_list


if __name__ == "__main__":
    # 裁切不需要的部分,返回处理后的
    data_list = get_main_area()
    img = cv2.imread(sample_edited_img_path)

    # 显示绘制好矩形的图片
    rect_set(img, data_list)
