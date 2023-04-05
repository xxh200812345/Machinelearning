from PIL import Image,ImageDraw,ImageFont
import pyocr
import pyocr.builders
import cv2
import numpy as np

#帳票の線を消す画像処理を追加してみました
def render_doc_text(file_path):

    # ツール取得
    pyocr.tesseract.TESSERACT_CMD = 'E:/Program Files/Tesseract-OCR/tesseract.exe'
    tools = pyocr.get_available_tools()
    tool = tools[0]

    # 画像取得
    img = cv2.imread(file_path, 0)
    
    # 必要に応じて画像処理 線を消す
    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_not(img)
    
    mlabel = cv2.connectedComponentsWithStats(img)
    data = np.delete(mlabel[2], 0, 0)

    new_image = np.zeros((img.shape[0], img.shape[1]))+255
    for i in range(mlabel[0]-1):
        if 0 < data[i][4] < 1000:
            new_image = np.where(mlabel[1] == i+1, 0, new_image)

    cv2.imwrite('sample_edited.png', new_image)

    img = Image.fromarray(new_image)

    # OCR
    builder = pyocr.builders.LineBoxBuilder(tesseract_layout=6)
    return tool.image_to_string(img, lang="jpn", builder=builder)

def rectSet(img,data_list):

    # 遍历每个矩形，绘制在图片上
    for data in data_list:
        rect = data.position
        x1, y1, x2, y2 = rect[0][0],rect[0][1],rect[1][0],rect[1][1]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 将图像从OpenCV格式转换为Pillow格式
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # 遍历每个矩形，绘制在图片上
    for data in data_list:
        rect = data.position
        x1, y1, x2, y2 = rect[0][0],rect[0][1],rect[1][0],rect[1][1]
        # 定义要绘制的文本和位置
        text = data.content
        # 定义文本样式
        font_path = "H:/vswork/AI/jpocr/NotoSansJP-Thin.otf"
        font_size = 16
        font_color = (0, 0, 255)

        # 加载日语字体
        font = ImageFont.truetype(font_path, font_size)

        # 在图像上绘制文本
        draw.text((x1, y1-23), text, font=font, fill=font_color)

    # 将图像从Pillow格式转换为OpenCV格式
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow('image with rectangles', img_cv)
    cv2.waitKey(0)



# OCR検知
data_list = render_doc_text('sample.png')
for d in data_list:
    print(d)
# 显示绘制好矩形的图片
rectSet(cv2.imread('sample_edited.png'),data_list)
