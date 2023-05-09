import os

# 护照对象
class Passport:
    def __init__(self, file_name):

        name, ext = os.path.splitext(file_name)
        # 护照文件名
        self.file_name = file_name
        # PDF转PNG后保存文件名
        self.pdf2png_file_name = f"{name}_pdf2png.png"
        # 第一次裁剪后文件名
        self.cut_file_name = f"{name}_cut.png"
        # 图像优化后文件名
        self.edited_file_name = f"{name}_edited.png"
        # 签名文件名
        self.sign_file_name = f"{name}_sign.png"
        # 图像识别后文件
        self.tessract_file_name = f"{name}_tessract.png"
        # OCR数据
        self.data_list = []
        # 文件类型
        self.ext = ext
        # 护照信息
        self.info={}
