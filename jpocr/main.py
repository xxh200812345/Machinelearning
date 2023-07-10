#!/bin/bash
# -*- coding: utf-8 -*-

from PassportOCRAbstraction import PassportOCRAbstraction

from passport import Passport
import os
import sys
import time

import glob
import tkinter
from tkinter import messagebox

os.chdir(os.path.abspath(os.path.dirname(__file__)))

# 配置文件数据组
config_options = {}

# 护照数组
passport_list = []

# 护照识别对象
ppoa = None

# 初始化设置
def init():
    # 读取配置文件
    global config_options, ppoa
    ppoa =  PassportOCRAbstraction()
    config_options = ppoa.config_options

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

    for index, PdfInPath in enumerate(pdf_files):
        # 记录开始时间
        start_time = time.time()

        print(f"正在处理： {index+1}， 共计 {len(pdf_files)}")
        ret = ppoa.passprocess(PdfInPath)

        # 记录函数结束时间
        end_time = time.time()
        # 计算函数运行时间
        run_time = end_time - start_time
        s_time += run_time

        print(f"处理完成，用时{round(run_time, 2)}秒\n")

    print(
        f"全部处理完成，一共{len(pdf_files)}件，总用时{round(s_time, 2)}秒，平均用时{round(s_time/len(pdf_files), 2)}秒"
    )
