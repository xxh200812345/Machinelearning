import cv2

# 读取原始图像并转换为灰度图像
img = cv2.imread("sample.jpg", cv2.IMREAD_COLOR)

def remove_small_height_regions(image,max_height,min_height):
    h, w = image.shape
    # 二值化图像
    _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

    # 对输入图像取反
    inverted_image = cv2.bitwise_not(binary)
    #膨胀操作
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(6,1))
    bin_clo=cv2.dilate(inverted_image,kernel2,iterations=2)
    cv2.imshow("bin_clo", bin_clo)
    cv2.waitKey(0)


    # 获取所有连通区域的标签
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_clo, connectivity=8
    )
    # 遍历每个连通区域，计算它们的高度
    for i in range(1, num_labels):
        height = stats[i, cv2.CC_STAT_HEIGHT]
        x1 = stats[i, cv2.CC_STAT_LEFT]
        y1 = stats[i, cv2.CC_STAT_TOP]
        x2 = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]
        y2 = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
        # 删除高度小于指定值的区域
        if height > min_height and height < max_height:
            binary[labels == i] = 0
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), -1)

    # # 将保留区域的标签映射回原始图像
    cv2.imshow("image with rectangles", img)
    cv2.waitKey(0)

image = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
remove_small_height_regions(image,15,9)
cv2.destroyAllWindows()
