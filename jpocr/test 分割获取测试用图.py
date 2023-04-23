import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyocr
import pyocr.builders
import cv2


# 目标图
sample_img_path = "sample.jpg"
sample_edited_img_path = "sample_edited.png"
sample_sign_img_path = "sample_sign.png"
pyocr.tesseract.TESSERACT_CMD = "E:/Program Files/Tesseract-OCR/tesseract.exe"
subimgs_path='cut_imgs/'

def get_original_img(path, mode):
    img = cv2.imread(path, mode)
    # 调整图像大小以便于处理
    img = cv2.resize(img, (600, 420))
    return img

# 图片二值化，把白色部分设置为透明
def binary_img_with_transparency(img, threshold=180):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 将二值化后的图像转换为4通道图像
    rgba = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGBA)

    # 将白色部分设置为透明
    rgba[:, :, 3] = np.where(gray == 255, 0, 255)
    return rgba

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

# 识别后输出成PNG小图片
def  output_subimg(img, data_list):
    # 定义边框宽度和颜色
    border_width = 8
    border_color = [255, 255, 255]  # 白色

    # 遍历每个矩形，绘制在图片上
    for data in data_list:
        rect = data.position
        x1, y1, x2, y2 = rect[0][0], rect[0][1], rect[1][0], rect[1][1]
        # 定义要绘制的文本和位置
        text = data.content

        img_part = img[ y1-2: y2+2 ,x1-2: x2+2]
        border_img = cv2.copyMakeBorder(img_part, border_width, border_width, border_width, border_width,
                                cv2.BORDER_CONSTANT, value=border_color)

        cv2.imwrite(f"{subimgs_path}{text}.png", border_img)


if __name__ == "__main__":
    # 裁切不需要的部分,返回处理后的
    data_list = ocr_by_key("word")
    img = get_original_img(sample_edited_img_path, cv2.IMREAD_COLOR)

    # 识别后输出成PNG小图片
    output_subimg(img, data_list)
