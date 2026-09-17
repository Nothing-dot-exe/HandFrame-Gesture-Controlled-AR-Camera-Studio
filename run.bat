@echo off
title HandFrame AI - Live Gesture-Controlled AI Style Transfer
echo ======================================================================
echo                 HandFrame AI - Camera Selector
echo ======================================================================
echo   [1] Default PC Webcam (Index 0)
echo   [2] Phone via DroidCam / Iriun / Camo USB (Index 1)
echo   [3] Phone via Wi-Fi (IP Webcam URL: http://...:8080/video)
echo ======================================================================
set /p choice="Select camera option [1, 2, 3] (or press ENTER for default): "

if "%choice%"=="2" (
    echo Launching with secondary camera (Phone / DroidCam / Iriun)...
    python "%~dp0app.py" --camera 1
) else if "%choice%"=="3" (
    echo.
    echo Make sure your phone and PC are on the same Wi-Fi.
    echo In your phone's IP Webcam app, tap 'Start Server'.
    set /p cam_url="Enter the URL shown on your phone (e.g. http://192.168.1.5:8080/video): "
    python "%~dp0app.py" --camera "%cam_url%"
) else (
    echo Launching with default PC webcam...
    python "%~dp0app.py" --camera 0
)

pause
