import visa_ocr

import pdf2img
import configparser
import os
import sys
import shutil
from mdocument import Document

import logging
from logging.handlers import RotatingFileHandler

os.chdir(os.path.abspath(os.path.dirname(__file__)))

class VisaOCRAbstraction:
    def __init__(self):
        self.config_options = self.config_readin()
        self.logger = self.getlogger()

    # 其他方法和属性
    def visaprocess(self, PdfInPath):
    
        self.logger.info(f"开始处理PDF :{PdfInPath}")
        
        try:
            # 使用PyMuPDF库将页面转换为图像
            pixs = pdf2img.pdf_page_to_image(f"{PdfInPath}" , self.logger)
        except Exception as e:
            self.logger.error(f"发生错误 {PdfInPath}: {str(e)}")
            return False
        
        if (len(pixs) < 2):
            self.logger.error("不存在第二页，找不到签证")
            return False

        # 第二页是签证，其他的是护照
        _visa = Document(PdfInPath , 1)

        output_dir = (
            self.config_options["OUTPUT_FOLDER_PATH"] + "/" + _visa.image_dir
        )

        # 保存图像
        pdf2img.save_pix2png(pixs[1], output_dir, _visa)
        visa_ocr.run(_visa, self.config_options, self.logger)

        return _visa.info

    # 读取配置文件
    def config_readin(self):
        config_options = {}

        config = configparser.ConfigParser()
        config.read("ocr_configs.ini", encoding="utf-8")

        # 获取某个 section 的所有键值对
        options = config.options("section")
        for option in options:
            value = config.get("section", option)

            config_options[option.upper()] = value

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
            config_options["OUTPUT_FOLDER_PATH"] + "/" + Document.image_dir
        )  # 如果文件夹不存在，则创建它
        os.makedirs(
            config_options["OUTPUT_FOLDER_PATH"] + "/" + Document.json_dir
        )  # 如果文件夹不存在，则创建它

        return config_options

    def getlogger(self):

        # Create a logger
        logger = logging.getLogger('visa_ocr')
        logger.setLevel(logging.DEBUG)  # or whichever level you prefer

        # Create a file handler
        log_file = 'visa_ocr.log'
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=3,encoding="utf-8")

        file_handler.setLevel(logging.DEBUG)
        if self.config_options["DEBUG"].lower() == "false":
            file_handler.setLevel(logging.INFO)

        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(file_handler)

        return logger
