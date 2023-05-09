import cv2

# 加载人脸检测器模型
face_cascade = cv2.HOGDescriptor()
face_cascade.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 加载图像
img = cv2.imread('/Users/xianhangxiao/vswork/Machinelearning/jpocr/output/【荻野有紀様】パスポートコピー_pdf2png.png')

# 检测人脸位置
faces, weights = face_cascade.detectMultiScale(img)

# 绘制矩形框
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 显示图像
cv2.imshow('img', img)
cv2.waitKey()
