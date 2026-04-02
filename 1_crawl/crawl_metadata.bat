@echo off
setlocal
cd /d %~dp0
echo [INFO] Starting Metadata Crawl (Kaggle Dataset mode)...
python logic\kaggle_metadata_reprocessor.py %*
pause
