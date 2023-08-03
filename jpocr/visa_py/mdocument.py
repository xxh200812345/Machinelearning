import os

class Document:
    def __init__(self, file_path, index):
        name, ext = os.path.splitext(os.path.basename(file_path))

        name += "_" + str(index).zfill(2)

        # 护照文件名
        self.file_name = name
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
        # 信息
        self.info = {}
        self.info["err_msg"] = ""


    # 图片存放位置
    image_dir = "images"
    # json存放位置
    json_dir = "jsons"
