import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyocr
import pyocr.builders
import cv2
import datetime


# 目标图
sample_img_path = "passport_imgs/sample.jpg"
sample_edited_img_path = "output/sample_edited.png"
sample_sign_img_path = "output/sample_sign.png"
pyocr.tesseract.TESSERACT_CMD = "/opt/homebrew/Cellar/tesseract/5.3.1/bin/tesseract"
subimgs_path='res/cut_imgs/'

def get_original_img(path, mode):
    img = cv2.imread(path, mode)
    # 调整图像大小以便于处理
    img = cv2.resize(img, (600, 420))
    return img


def ocr_by_key(key):
    # ツール取得
    tools = pyocr.get_available_tools()
    tool = tools[0]

    # グレースケールでイメージを読み込み
    img = cv2.imread(sample_edited_img_path, cv2.IMREAD_GRAYSCALE)

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

# 识别后输出成PNG小图片
def  output_subimg(img, data_list):
    # 定义边框宽度和颜色
    border_width = 8
    border_color = [255, 255, 255]  # 白色

    now = datetime.datetime.now()
    #time
    mtime=now.strftime("%Y%m%d%H%M%S")

    # 遍历每个矩形，绘制在图片上
    i=0
    for data in data_list:
        rect = data.position
        x1, y1, x2, y2 = rect[0][0], rect[0][1], rect[1][0], rect[1][1]
        # 定义要绘制的文本和位置
        text = data.content

        img_part = img[ y1-2: y2+2 ,x1-2: x2+2]
        border_img = cv2.copyMakeBorder(img_part, border_width, border_width, border_width, border_width,
                                cv2.BORDER_CONSTANT, value=border_color)

        cv2.imwrite(f"{subimgs_path}rt_{mtime}_{str(i).zfill(3)}_{text}.png", border_img)
        i+=1


if __name__ == "__main__":
    # 裁切不需要的部分,返回处理后的
    data_list = ocr_by_key("digits")
    img = get_original_img(sample_edited_img_path, cv2.IMREAD_COLOR)

    # 识别后输出成PNG小图片
    output_subimg(img, data_list)
