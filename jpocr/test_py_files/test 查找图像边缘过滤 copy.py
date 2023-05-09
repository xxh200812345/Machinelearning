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


paths=get_file_paths("res/护照定位测试文件")

# 读取图像
image = cv2.imread(paths[3])

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 去噪
blur = cv2.GaussianBlur(gray, (5, 5), 0)

def a():

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
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    # 绘制检测到的直线
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 显示结果图像
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

a()

# 二值化
threshold_value = 245
_, binary = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)

# 轮廓提取
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 根据面积排序
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# 获取面积最大的三个矩形
max_rectangles = []
for contour in contours:
    # 计算轮廓的外接矩形
    x, y, w, h = cv2.boundingRect(contour)

    # 计算矩形的面积
    area = w * h

    # 如果当前矩形面积比最小的矩形面积要大，就将其加入到列表中，并删除最小的矩形
    if len(max_rectangles) < 2 or area > max_rectangles[0][0]:
        max_rectangles.append((area, x, y, w, h))
        max_rectangles = sorted(max_rectangles, key=lambda r: r[0], reverse=True)[:3]

# 遍历轮廓并绘制边缘
_,x, y, w, h = max_rectangles[1]
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 显示图像
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
