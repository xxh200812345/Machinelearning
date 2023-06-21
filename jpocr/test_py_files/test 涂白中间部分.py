import cv2
import numpy as np

# 把图片的中间部分涂白
def fill_middle_with_white(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 获取图像的尺寸
    height, width = image.shape[:2]

    # 计算矩形区域的左上角和右下角坐标
    top_left = (0, int(height*0.25))
    bottom_right = (width, int(height*0.65))

    # 创建与原始图像大小相同的空白图像
    mask = np.zeros_like(image)

    # 在空白图像上绘制矩形区域为白色
    cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255), cv2.FILLED)

    # 将矩形区域应用到原始图像上
    result = cv2.bitwise_or(image, mask)

    # 保存结果图像
    cv2.imwrite(output_path, result)

# 调用函数并传入图像路径和输出路径
fill_middle_with_white(r"H:\vswork\Machinelearning\jpocr\output\images\aaa.png", "output_image.png")
