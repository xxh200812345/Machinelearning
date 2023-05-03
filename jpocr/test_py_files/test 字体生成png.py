import random
from PIL import Image, ImageDraw, ImageFont
import string
from collections import Counter
import datetime

# 设置图片大小和背景颜色
image_width = 13
image_height = 35
background_color = (255, 255, 255)

# 获取字体文件列表
font_name = "res/OCRB.ttf"
subimgs_path = "res/cut_imgs/"

# 生成大写英文字母和数字的字符集
chars = string.ascii_uppercase + string.ascii_lowercase + string.digits + "><"
#chars = "JPNFH"

# 设置字体大小和位置
font_size = 16
x = 10
y = 8

#以下是一个Python函数，用于生成指定重复次数的字母表字符串，打乱顺序，然后把这个字符串分割成指定范围长度的字符串数组
def generate_random_strings(count, min_len, max_len,alphabet):
    # 生成指定重复次数的字母表字符串
    str_list = list(alphabet * count)
    random.shuffle(str_list)
    random_string = "".join(str_list)

    # 将字符串分割成指定范围长度的字符串数组
    result = []
    start_index = 0
    while start_index < len(random_string):
        end_index = start_index + random.randint(min_len, max_len)
        if end_index >= len(random_string):
            end_index = len(random_string)
        result.append(random_string[start_index:end_index])
        start_index = end_index

    return result

#统计字符串数组中，各字符出现的频率
def count_chars(strings):
    counter = Counter()
    for s in strings:
        counter.update(s)
    return counter


if __name__ == "__main__":
    result = generate_random_strings(2, 15, 20, chars)

    now = datetime.datetime.now()

    #time
    mtime=now.strftime("%Y%m%d%H%M%S")

    #统计字符串数组中，各字符出现的频率
    char_count = count_chars(result)

    for char, count in char_count.items():
        print(f"{char}: {count}")

    with open(f"{subimgs_path}random_str_data", 'w') as output_file:

        for i in range(len(result)):

            # 创建一个空白图片
            image = Image.new(
                "RGB", (image_width * len(result[i]) + 15, image_height), background_color
            )

            # 加载字体文件并创建画笔
            font = ImageFont.truetype(font_name, font_size)
            draw = ImageDraw.Draw(image)

            # 在图片上写入文字
            draw.text((x, y), result[i], fill=(0, 0, 0), font=font)

            # 保存图片
            image.save(f"{subimgs_path}rt_{mtime}_{str(i).zfill(3)}_{result[i]}.png")
            
            output_file.write(result[i]+'\n')

    

