REM Wait for user input of the Tesseract-OCR folder path without the final
REM Example: E:\Program Files\Tesseract-OCR
set /p "OCR_path=Please enter the Tesseract-OCR folder path:"

REM Print the input Tesseract-OCR folder path
echo The path you entered is: %OCR_path%

REM Jump to the current folder of this file
pushd "%~dp0"

REM Execute other commands or operations
echo Current working directory: %CD%

REM Automatically set whitelist
copy /Y "res\_my_word" "%OCR_path%\tessdata\configs"

pause

REM Set custom models

REM Copy num_1.traineddata model file to tessdata folder
copy /Y "res\num_1.traineddata" "%OCR_path%\tessdata"

REM Copy eng.traineddata model file to tessdata folder
copy /Y "res\eng.traineddata" "%OCR_path%\tessdata"

REM Copy jpn.traineddata model file to tessdata folder
copy /Y "res\jpn.traineddata" "%OCR_path%\tessdata"

REM Check if installed correctly: eng jpn num_1
tesseract --list-langs

pause

REM Configure virtual environment
set VIRTUALENV_PATH=ai_jpocr_venv

if exist %VIRTUALENV_PATH% (
echo Virtual environment already exists
) else (
    echo Virtual environment does not exist, creating a new virtual environment...

    python -m venv "%VIRTUALENV_PATH%"
    
)
call %VIRTUALENV_PATH%\Scripts\activate

pip install -r requirements.txt

REM Installation completed

pause


