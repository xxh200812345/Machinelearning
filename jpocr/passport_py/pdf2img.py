
import fitz
import os
from passport import Passport


# 遍历PDF页面并转换为图像
def pdf_page_to_image(pdf_path, dpi=300):
    # 打开PDF文件
    pdf_doc = fitz.open(pdf_path)

    # PDF第一页转换为图像
    page = pdf_doc[0]

    # 使用PyMuPDF库将页面转换为图像
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))

    return pix

# 保存pix为PNG
def save_pix2png(pix, dir, passport: Passport):
    pix.save(f"{dir}/{passport.pdf2png_file_name}")
