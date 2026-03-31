@echo off
setlocal

cd /d D:\project
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

python scripts\build_audio_dataset.py ^
  --metadata-csv data\mbti_cnn_metadata.csv ^
  --duration 20 ^
  --download-workers 32 ^
  --ffmpeg-workers 10 ^
  --progress-every 25 ^
  --min-size-bytes 180000 ^
  --sleep-interval-requests 0.75 ^
  --sleep-interval 2 ^
  --max-sleep-interval 5

set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Crawl finished with exit code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
