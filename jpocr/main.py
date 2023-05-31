#!/bin/bash
# -*- coding: utf-8 -*-

import passport_ocr

import pdf2img
import configparser
from passport import Passport
import os
import shutil

os.chdir(os.path.abspath(os.path.dirname(__file__)))

# 配置文件数据组
config_options = {}

# 护照数组
passport_list = []


# 读取配置文件
def config_readin():
    global config_options

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


# 初始化设置
def init():
    # 读取配置文件
    config_readin()

    if not os.path.exists(config_options["OUTPUT_FOLDER_PATH"]):
        os.makedirs(config_options["OUTPUT_FOLDER_PATH"])  # 如果文件夹不存在，则创建它
        os.makedirs(config_options["OUTPUT_FOLDER_PATH"] + "/" + Passport.image_dir)  # 如果文件夹不存在，则创建它
    else:
        shutil.rmtree(config_options["OUTPUT_FOLDER_PATH"])  # 如果文件夹已经存在，则清空它
        os.makedirs(config_options["OUTPUT_FOLDER_PATH"])  # 然后再创建它
        os.makedirs(config_options["OUTPUT_FOLDER_PATH"] + "/" + Passport.image_dir)  # 然后再创建它

    # 新建输出数据文件
    output_data_file = config_options["OUTPUT_FOLDER_PATH"]+ "/" + Passport.data_json_name
    open(output_data_file, "w", encoding="utf-8")

    # 新建文字信息拆分后数据文件夹
    os.makedirs(config_options["OUTPUT_FOLDER_PATH"] + "/text_imgs")


if __name__ == "__main__":
    # 初始化设置
    init()

    # 被识别图像所在文件夹
    passport_imgs_dir = config_options["PASSPORT_IMAGES_FOLDER_PATH"]

    # 被识别PDF所在文件夹
    passport_pdfs_dir = config_options["PASSPORT_PDFS_FOLDER_PATH"]

    # 识别后输出文件夹
    output_dir = config_options["OUTPUT_FOLDER_PATH"] + '/' + Passport.image_dir

    paths = []
    if os.path.exists(passport_imgs_dir):
        paths += os.listdir(passport_imgs_dir)
    if os.path.exists(passport_pdfs_dir):
        paths += os.listdir(passport_pdfs_dir)

    # 遍历文件夹下所有文件
    for file_name in paths:
        if file_name.endswith(".pdf"):

            # if("中島敏壽様　パスポート" not in file_name):
            #     continue

            print(f"开始处理:{passport_pdfs_dir}/{file_name}")

            passport = Passport(file_name)

            # 使用PyMuPDF库将页面转换为图像
            pix = pdf2img.pdf_page_to_image(f"{passport_pdfs_dir}/{file_name}")
            # 保存图像
            pdf2img.save_pix2png(pix, passport.pdf2png_file_name, output_dir)
            passport_ocr.run(passport, config_options)

            passport_list.append(passport)

        # 如果是图片文件，则打印文件名
        if (
            file_name.endswith(".jpg")
            or file_name.endswith(".jpeg")
            or file_name.endswith(".png")
        ):
            print(f"开始处理:{passport_imgs_dir}/{file_name}")
            passport = Passport(file_name)
            passport_ocr.run(passport,config_options)
            passport_list.append(passport)

    # 识别后数据输出到文本文件中
    passport_ocr.output_data2text_file(passport_list, config_options)
