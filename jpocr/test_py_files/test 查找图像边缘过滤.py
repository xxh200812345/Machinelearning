import cv2
import numpy as np
import os

def get_file_paths(folder_path):
    """
    获取文件夹中所有文件路径
    :param folder_path: 文件夹路径
    :return: 文件路径列表
    """
    file_paths = []  # 文件路径列表
    # 获取文件夹中所有文件名
    file_names = os.listdir(folder_path)
    # 遍历所有文件名
    for file_name in file_names:
        # 拼接文件路径
        file_path = os.path.join(folder_path, file_name)
        # 判断文件是否为文件夹
        if os.path.isdir(file_path):
            # 如果是文件夹，则递归调用该函数
            sub_file_paths = get_file_paths(file_path)
            file_paths.extend(sub_file_paths)
        else:
            # 如果是文件，则将文件路径添加到文件路径列表中
            file_paths.append(file_path)
    return file_paths


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

#遍历轮廓并绘制边缘
def get_lines(blur):

    # 使用Canny算法查找边缘
    edges = cv2.Canny(blur, 100, 200)

    # 对边缘进行平滑处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 轮廓提取
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历轮廓并绘制边缘
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # # 显示图像
    # cv2.imshow("Image", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

        # 对边缘进行直线检测
    # lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    # # 绘制检测到的直线
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 显示结果图像
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":

    paths=get_file_paths("res/护照定位测试文件")


    for path in paths:
        # 读取图像
        image = cv2.imread(path)

        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        binary = color_scale_display(blurred, 160, 220, 0.79)

        # # 二值化
        # threshold_value = 245
        # _, binary = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

        get_lines(binary)

        # 显示图像
        # cv2.imshow("Image", binary)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()