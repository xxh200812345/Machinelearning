import pytesseract
from PIL import Image
import cv2
import numpy as np

# 获取识别文字的方向，并旋转图片，只能识别90 180 270
def rotate_image_with_white_bg(image):
    # 获取识别文字的位置信息
    data = pytesseract.image_to_osd(image)
    print(data)
    # 从位置信息中提取文字方向
    lines = data.split('\n')
    angle = None
    for line in lines:
        if line.startswith('Orientation in degrees:'):
            angle = float(line.split(':')[1].strip())
            break

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
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), borderValue=(255, 255, 255))

    return rotated_image


# 读取图像
image = Image.open('passport_imgs/荻野様 180度.png')
image=cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
cv2.imshow("", rotate_image_with_white_bg(image))
cv2.waitKey(0)  # 等待用户按下键盘上的任意键
cv2.destroyAllWindows()  # 关闭所有cv2.imshow窗口