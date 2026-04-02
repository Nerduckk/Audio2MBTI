@echo off
setlocal
cd /d %~dp0
echo [INFO] Starting Audio Downloader (YouTube Search mode)...
echo [INFO] Reading from: data\mbti_cnn_metadata.csv
python logic\build_audio_dataset.py --metadata-csv ..\data\mbti_cnn_metadata.csv --duration 20 --per-label-limit 100 %*
pause
