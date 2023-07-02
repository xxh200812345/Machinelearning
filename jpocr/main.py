#!/bin/bash
# -*- coding: utf-8 -*-

import passport_ocr

import pdf2img
import configparser
from passport import Passport
import os
import sys
import shutil
import time

import glob
import tkinter
from tkinter import messagebox

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

    print(f"len(sys.argv): {sys.argv}")

    if len(sys.argv) == 3:
        config_options["PASSPORT_PDFS_FOLDER_PATH"] = sys.argv[1]
        config_options["OUTPUT_FOLDER_PATH"] = sys.argv[2]

    if len(sys.argv) != 1 and len(sys.argv) != 3:
        error_exit("请输入命令：main.py input_path output_path")

    # 传入文件夹地址时去除末尾的反斜杠符号（\）
    config_options["PASSPORT_PDFS_FOLDER_PATH"] = config_options[
        "PASSPORT_PDFS_FOLDER_PATH"
    ].rstrip(os.sep)
    config_options["OUTPUT_FOLDER_PATH"] = config_options["OUTPUT_FOLDER_PATH"].rstrip(
        os.sep
    )


# 初始化设置
def init():
    # 读取配置文件
    config_readin()

    if not os.path.exists(config_options["OUTPUT_FOLDER_PATH"]):
        os.makedirs(config_options["OUTPUT_FOLDER_PATH"])  # 如果文件夹不存在，则创建它
        os.makedirs(
            config_options["OUTPUT_FOLDER_PATH"] + "/" + Passport.image_dir
        )  # 如果文件夹不存在，则创建它
    else:
        shutil.rmtree(config_options["OUTPUT_FOLDER_PATH"])  # 如果文件夹已经存在，则清空它
        os.makedirs(config_options["OUTPUT_FOLDER_PATH"])  # 然后再创建它
        os.makedirs(
            config_options["OUTPUT_FOLDER_PATH"] + "/" + Passport.image_dir
        )  # 然后再创建它

    # 新建文字信息拆分后数据文件夹
    os.makedirs(config_options["OUTPUT_FOLDER_PATH"] + "/text_imgs")


def error_exit(info_msg):
    # 创建Tkinter根窗口
    root = tkinter.Tk()
    # 隐藏根窗口
    root.withdraw()
    print(info_msg)
    messagebox.showinfo("提示", info_msg)
    root.quit()  # 关闭应用程序
    sys.exit()


if __name__ == "__main__":
    # 初始化设置
    init()

    # 被识别图像所在文件夹
    passport_imgs_dir = config_options["PASSPORT_IMAGES_FOLDER_PATH"]

    # 被识别PDF所在文件夹
    passport_pdfs_dir = config_options["PASSPORT_PDFS_FOLDER_PATH"]

    # 识别后输出图片文件夹
    output_dir = config_options["OUTPUT_FOLDER_PATH"] + "/" + Passport.image_dir

    # 查询文件夹中是否存在pdf文件
    pdf_files = glob.glob(os.path.join(passport_pdfs_dir, "*.pdf"))
    if not pdf_files:
        error_exit("文件夹中没有PDF文件！")

    paths = []
    if os.path.exists(passport_imgs_dir):
        paths += os.listdir(passport_imgs_dir)
    if os.path.exists(passport_pdfs_dir):
        paths += os.listdir(passport_pdfs_dir)

    # 遍历文件夹下所有文件
    pdf_paths = []

    for file_name in paths:
        if file_name.endswith(".pdf"):
            pdf_paths.append(file_name)

        # # 如果是图片文件，则打印文件名
        # if (
        #     file_name.endswith(".jpg")
        #     or file_name.endswith(".jpeg")
        #     or file_name.endswith(".png")
        # ):
        #     print(f"开始处理JPG:{passport_imgs_dir}/{file_name}")
        #     passport = Passport(file_name)
        #     passport_ocr.run(passport, config_options)
        #     passport_list.append(passport)

    # 处理时间
    s_time = 0

    for index, file_name in enumerate(pdf_files):
        # 记录开始时间
        start_time = time.time()

        passport = Passport(file_name)
        print(f"开始处理PDF ({index+1}/{len(pdf_files)}):{file_name}")

        # 使用PyMuPDF库将页面转换为图像
        pix = pdf2img.pdf_page_to_image(f"{file_name}")
        # 保存图像
        pdf2img.save_pix2png(pix, output_dir, passport)

        passport_ocr.run(passport, config_options)

        passport_list.append(passport)

        # 记录函数结束时间
        end_time = time.time()
        # 计算函数运行时间
        run_time = end_time - start_time
        s_time += run_time

        print(f"处理完成，用时{round(run_time, 2)}秒\n")

    print(
        f"全部处理完成，一共{len(pdf_files)}件，总用时{round(s_time, 2)}秒，平均用时{round(s_time/len(pdf_files), 2)}秒"
    )
