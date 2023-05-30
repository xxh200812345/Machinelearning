chcp 65001 > nul

REM 等待用户输入Tesseract-OCR的文件夹路径，不要最后带\
REM 比如：E:\Program Files\Tesseract-OCR
set /p "OCR_path=请输入Tesseract-OCR的文件夹路径："

REM 打印输入的Tesseract-OCR的文件夹路径
echo 你输入的路径是：%OCR_path%

REM 跳转到当前文件的文件夹
pushd "%~dp0"

REM 执行其他命令或操作
echo 当前工作目录：%CD%

REM 自动设置白名单
copy /Y "res\_my_word" "%OCR_path%\tessdata\configs\"

pause

REM 设置自定义模型

REM 复制 num_1.traineddata 模型文件到 tessdata 文件夹
copy /Y "jpocr\res\num_1.traineddata" "%OCR_path%\tessdata\"

REM 复制 eng.traineddata 模型文件到 tessdata 文件夹
copy /Y "jpocr\res\eng.traineddata" "%OCR_path%\tessdata\"

REM 复制 jpn.traineddata 模型文件到 tessdata 文件夹
copy /Y "jpocr\res\jpn.traineddata" "%OCR_path%\tessdata\"

REM 查看是否正确安装：`eng` `jpn` `num_1`
tesseract --list-langs

pause

REM 配置虚拟环境
set VIRTUALENV_PATH=ai_jpocr_venv

if exist %VIRTUALENV_PATH% (
    echo Virtual environment already exists
) else (
    echo Virtual environment does not exist, creating a new virtual environment...

    python -m venv "%VIRTUALENV_PATH%"
    
)
call %VIRTUALENV_PATH%\Scripts\activate

pip install -r requirements.txt

REM 安装完毕

pause


