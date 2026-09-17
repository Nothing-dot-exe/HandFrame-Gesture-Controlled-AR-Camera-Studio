@echo off
title HandFrame AI - Phone Wi-Fi Stream
echo ========================================================
echo   Make sure phone and PC are on the same Wi-Fi.
echo   Open 'IP Webcam' on phone and tap 'Start Server'.
echo ========================================================
set /p cam_url="Enter phone URL (e.g. http://192.168.1.5:8080/video): "
python "%~dp0app.py" --camera "%cam_url%"
pause
