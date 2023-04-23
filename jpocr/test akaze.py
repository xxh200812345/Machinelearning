import cv2
import numpy as np

# 读取原始图像并转换为灰度图像
img = cv2.imread('sample.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 应用二值化阈值分割
ret, img_adaptive = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# 初始化AKAZE
akaze = cv2.AKAZE_create()

# 检测关键点和描述符
keypoints, descriptors = akaze.detectAndCompute(img_adaptive, None)

# # 绘制关键点的边界
# for kp in keypoints:
#     x, y = np.int32(kp.pt)
#     r = np.int32(kp.size/2)
#     cv2.rectangle(img, (x-r, y-r), (x+r, y+r), (0, 255, 0), 2)


# 显示结果
img = cv2.drawKeypoints(img_adaptive, keypoints, None)
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()