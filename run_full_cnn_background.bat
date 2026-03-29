@echo off
setlocal

chdir /d D:\project

set PIPELINE_CRAWL_BATCH_SIZE=10
set PIPELINE_SPOTIFY_RPS=0.75
set PIPELINE_PLAYLIST_DELAY_MIN=1.0
set PIPELINE_PLAYLIST_DELAY_MAX=2.0
set PIPELINE_PER_LABEL_LIMIT=15
set PIPELINE_TOTAL_LIMIT=240
set PIPELINE_AUDIO_DURATION=20
set PIPELINE_MIN_AUDIO_DURATION=12
set PIPELINE_MIN_AUDIO_SIZE_BYTES=180000
set PIPELINE_MIN_TRAIN_SAMPLES=64
set PIPELINE_MIN_LABEL_COVERAGE=8

echo starting background pipeline...
start "audio2mbti-background" /min /low cmd /c "cd /d D:\project && call run_full_cnn_pipeline.bat >> logs\cnn_background.log 2>&1"
echo background pipeline started. log: logs\cnn_background.log
exit /b 0
