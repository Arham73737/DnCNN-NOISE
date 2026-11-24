@echo off
cd /d "%~dp0"
set PYTHONPATH=%CD%\venv\Lib\site-packages
echo Starting Image Denoising App at http://127.0.0.1:5000 ...
python app.py
pause
