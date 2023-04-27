# windows部署手册

## 安装tessract
https://www.jianshu.com/p/f7cb0b3f337a
根据教程下载安装windows版本

命令行使用 `tesseract --list-langs` 命令可查看当前软件支持的语言

## 部署OCR程序
从github链接clone项目到windows 
https://github.com/xxh200812345/Machinelearning.git

### 配置虚拟环境

    python -m venv ai_jpocr_venv
    ai_jpocr_venv\Scripts\activate
    pip install -r requirements.txt

### 设置安装tessract的应用路径
### 设置白名单
### 设置自定义模型
### 运行
## 测试
### 输入图片标准说明
### 输出结果以及说明
