import visa_ocr

import pdf2img
import configparser
from mvisa import Visa
import os
import sys
import shutil
from mdocument import Document


class VisaOCRAbstraction:
    def __init__(self):
        self.config_options = self.config_readin()

    # 其他方法和属性
    def passprocess(self, PdfInPath):

        print(f"开始处理PDF :{PdfInPath}")

        # 使用PyMuPDF库将页面转换为图像
        pixs = pdf2img.pdf_page_to_image(f"{PdfInPath}")

        # 第二页是签证，其他的是护照
        _visa = Visa(PdfInPath , 1)

        output_dir = (
            self.config_options["OUTPUT_FOLDER_PATH"] + "/" + _visa.image_dir
        )

        # 保存图像
        pdf2img.save_pix2png(pixs[1], output_dir, _visa)
        visa_ocr.run(_visa, self.config_options)

        return _visa.info

    # 读取配置文件
    def config_readin(self):
        config_options = {}

        config = configparser.ConfigParser()
        config.read("ocr_configs.ini", encoding="utf-8")

        # 获取整个配置文件的所有 section
        sections = config.sections()
        print(f"Sections: {sections}")

        # 获取某个 section 的所有键值对
        options = config.options("section")
        for option in options:
            value = config.get("section", option)

            config_options[option.upper()] = value

        print(f"len(sys.argv): {sys.argv}")

        # 传入文件夹地址时去除末尾的反斜杠符号（\）
        config_options["PDFS_FOLDER_PATH"] = config_options[
            "PDFS_FOLDER_PATH"
        ].rstrip(os.sep)
        config_options["OUTPUT_FOLDER_PATH"] = config_options[
            "OUTPUT_FOLDER_PATH"
        ].rstrip(os.sep)

        if not os.path.exists(config_options["OUTPUT_FOLDER_PATH"]):
            os.makedirs(config_options["OUTPUT_FOLDER_PATH"])  # 如果文件夹不存在，则创建它
        else:
            shutil.rmtree(config_options["OUTPUT_FOLDER_PATH"])  # 如果文件夹已经存在，则清空它
            os.makedirs(config_options["OUTPUT_FOLDER_PATH"])  # 然后再创建它

        os.makedirs(
            config_options["OUTPUT_FOLDER_PATH"] + "/" + Visa.image_dir
        )  # 如果文件夹不存在，则创建它
        os.makedirs(
            config_options["OUTPUT_FOLDER_PATH"] + "/" + Visa.json_dir
        )  # 如果文件夹不存在，则创建它

        return config_options

