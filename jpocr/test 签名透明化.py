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

# 图片二值化，把白色部分设置为透明
def binary_img_with_transparency(img, threshold=180):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # 将二值化后的图像转换为4通道图像
    rgba = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGBA)

    # 将白色部分设置为透明
    rgba[:, :, 3] = np.where(thresh == 255, 0, 255)
    return rgba

if __name__ == "__main__":
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