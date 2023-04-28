import passport_ocr
import configparser
from passport import Passport
import os

# 配置文件数据组
config_options={}

# 护照数组
passport_list=[]

# 读取配置文件
def config_readin():
    global config_options

    config = configparser.ConfigParser()
    config.read('ocr_configs.ini',encoding='utf-8')

    # 获取整个配置文件的所有 section
    sections = config.sections()
    print(f"Sections: {sections}")

    # 获取某个 section 的所有键值对
    options = config.options('section')
    for option in options:
        value = config.get('section', option)
        print(f"{option}: {value}")
        
        config_options[option.upper()] = value

# 初始化设置
def init():
    # 读取配置文件
    config_readin()
    
    # 护照OCR模块初始化
    #passport_ocr.init(config_options)

if __name__ == "__main__":

    # 初始化设置
    init()

    # 被识别图像所在文件夹
    passport_imgs_dir = config_options["PASSPORT_IMAGES_FOLDER_PATH"]

    # 清空输出数据文件
    output_data_file = config_options["OUTPUT_FOLDER_PATH"]+"/data.txt"

    # 打开文件，将文件指针移动到文件的开头
    with open(output_data_file, "w") as f:
        f.truncate(0)  # 清空文件内容

    # 遍历文件夹下所有文件
    for file_name in os.listdir(passport_imgs_dir):
        # 如果是图片文件，则打印文件名
        if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
            print(f"开始处理:{passport_imgs_dir}/{file_name}")
            passport = Passport(file_name)
            passport_ocr.run(passport,config_options)
            passport_list.append(passport)

