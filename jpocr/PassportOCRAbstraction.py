import passport_ocr

import pdf2img
import configparser
from passport import Passport
import os
import sys
import json
import shutil


class PassportOCRAbstraction:
    def __init__(self):
        self.config_options = self.config_readin()
        self.passport = None

    # 其他方法和属性
    def passprocess(self, PdfInPath):
        self.passport = Passport(PdfInPath)

        passport = self.passport

        print(f"开始处理PDF :{PdfInPath}")

        output_dir = (
            self.config_options["OUTPUT_FOLDER_PATH"] + "/" + Passport.image_dir
        )

        # 使用PyMuPDF库将页面转换为图像
        pix = pdf2img.pdf_page_to_image(f"{PdfInPath}")
        # 保存图像
        pdf2img.save_pix2png(pix, output_dir, passport)

        passport_ocr.run(passport, self.config_options)

        return passport.info

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
        config_options["PASSPORT_PDFS_FOLDER_PATH"] = config_options[
            "PASSPORT_PDFS_FOLDER_PATH"
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
            config_options["OUTPUT_FOLDER_PATH"] + "/" + Passport.image_dir
        )  # 如果文件夹不存在，则创建它
        os.makedirs(
            config_options["OUTPUT_FOLDER_PATH"] + "/" + Passport.json_dir
        )  # 如果文件夹不存在，则创建它

        return config_options

    # 识别后数据输出到文本文件中
    def output_data2text_file(self, passport):
        passport = self.passport

        output_data_file = (
            self.config_options["OUTPUT_FOLDER_PATH"]
            + "/"
            + Passport.json_dir
            + "/"
            + passport.file_name
            + ".json"
        )
        # 打开文件，将文件指针移动到文件的末尾
        with open(output_data_file, "a", encoding="utf-8") as f:
            json.dump(passport.info, f, ensure_ascii=False)
