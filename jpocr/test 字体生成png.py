import random
from PIL import Image, ImageDraw, ImageFont
import string
from collections import Counter

# 设置图片大小和背景颜色
image_width = 13
image_height = 35
background_color = (255, 255, 255)

# 获取字体文件列表
font_name = "OCRB.ttf"
subimgs_path = "cut_imgs/"

# 生成大写英文字母和数字的字符集
chars = string.ascii_uppercase + string.ascii_lowercase + string.digits + "><"

# 设置字体大小和位置
font_size = 16
x = 10
y = 8


# 为了使生成的字符串中字母的使用次数相对平均，我们可以先生成一个列表，其中每个字母出现的次数相等，然后将其分割成长度为 8 到 10 的子列表
def generate_strings(n, min_len, max_len, chars):
    alphabet = list(chars)
    repeats = n * max_len // len(alphabet) + 1
    extended_alphabet = alphabet * repeats
    random.shuffle(extended_alphabet)

    total_len = n * max_len
    index = 0
    result = []
    for _ in range(n):
        length = random.randint(min_len, max_len)
        if index + length > total_len:
            length = total_len - index
        current_alphabet = extended_alphabet[index : index + length]
        random.shuffle(current_alphabet)
        result.append("".join(current_alphabet))
        index += length

    return result

#统计字符串数组中，各字符出现的频率
def count_chars(strings):
    counter = Counter()
    for s in strings:
        counter.update(s)
    return counter


if __name__ == "__main__":
    result = generate_strings(50, 12, 13, chars)

    #统计字符串数组中，各字符出现的频率
    # char_count = count_chars(result)

    # for char, count in char_count.items():
    #     print(f"{char}: {count}")

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
        print(result[i])

        # 保存图片
        image.save(f"{subimgs_path}rt_{str(i).zfill(3)}.png")
