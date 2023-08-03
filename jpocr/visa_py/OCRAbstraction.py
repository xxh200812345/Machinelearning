import passport_ocr
import visa_ocr

import pdf2img
import configparser
from passport import Passport
from mvisa import Visa
import os
import sys
import shutil


class OCRAbstraction:
    def __init__(self):
        self.config_options = self.config_readin()
        self.pages = []

    # 其他方法和属性
    def passprocess(self, PdfInPath):
        pages = self.pages

        print(f"开始处理PDF :{PdfInPath}")

        # 使用PyMuPDF库将页面转换为图像
        pixs = pdf2img.pdf_page_to_image(f"{PdfInPath}")

        # 第二页是签证，其他的是护照
        for i, pix in enumerate(pixs):
            if i != 1 :
                pages.append(Passport(PdfInPath , i))
            else :
                pages.append(Visa(PdfInPath , i))

        for i, page in enumerate(pages):

            output_dir = (
                self.config_options["OUTPUT_FOLDER_PATH"] + "/" + page.image_dir
            )

            if i != 1 :
                # 保存图像
                continue
                pdf2img.save_pix2png(pixs[i], output_dir, page)
                passport_ocr.run(page, self.config_options)

            else :
                # 保存图像
                pdf2img.save_pix2png(pixs[i], output_dir, page)
                visa_ocr.run(page, self.config_options)

        info_array = [pages.info for page in pages]
        return info_array

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
            config_options["OUTPUT_FOLDER_PATH"] + "/" + Passport.image_dir
        )  # 如果文件夹不存在，则创建它
        os.makedirs(
            config_options["OUTPUT_FOLDER_PATH"] + "/" + Passport.json_dir
        )  # 如果文件夹不存在，则创建它

        return config_options

