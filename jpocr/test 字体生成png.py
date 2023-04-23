import random
from PIL import Image, ImageDraw, ImageFont
import string

# 设置图片大小和背景颜色
image_width = 12
image_height = 25
background_color = (255, 255, 255)

# 获取字体文件列表
font_name = "OCRB.ttf"
subimgs_path = "cut_imgs/"

# 生成大写英文字母和数字的字符集
chars = string.ascii_uppercase + string.digits + "<"

# 设置字体大小和位置
font_size = 16
x = 5
y = 5

for i in range(30):
    # 生成随机字符串
    k_random = random.randint(4, 8)
    random_string = "".join(random.choices(chars, k=k_random))
    print(random_string)

    # 创建一个空白图片
    image = Image.new(
        "RGB", (image_width * k_random + 10, image_height), background_color
    )

    # 加载字体文件并创建画笔
    font = ImageFont.truetype(font_name, font_size)
    draw = ImageDraw.Draw(image)

    # 在图片上写入文字
    draw.text((x, y), random_string, fill=(0, 0, 0), font=font)

    # 保存图片
    image.save(f"{subimgs_path}rt_{str(i).zfill(3)}.png")
