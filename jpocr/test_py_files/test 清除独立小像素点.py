import cv2
import numpy as np

# 读取图像
image = cv2.imread(r"H:\vswork\Machinelearning\jpocr\output\images\aaa.png")

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def find_and_draw_objects(gray):

    # 二值化处理
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制矩形轮廓
    new_contours=[]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < 400 and h < 20:
           new_contours.append(contour) 

    # 创建与原始图像相同大小的空白掩码
    mask = np.zeros_like(image)

    # 填充轮廓区域为白色
    cv2.drawContours(mask, new_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    result = cv2.bitwise_or(image, mask)

    return result

image = find_and_draw_objects(gray)

# 显示和保存结果
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
