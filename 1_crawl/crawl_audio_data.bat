@echo off
setlocal
cd /d %~dp0
echo ==========================================
echo MBTI Music Intelligence: Audio Collection
echo ==========================================
echo [1] Run Metadata Crawl (Kaggle Dataset)
echo [2] Run Audio Downloader (YouTube)
echo [3] Exit
echo ==========================================
set /p choice="Enter choice [1-3]: "

if "%choice%"=="1" call crawl_metadata.bat
if "%choice%"=="2" call crawl_audio.bat
if "%choice%"=="3" exit /b 0

pause
