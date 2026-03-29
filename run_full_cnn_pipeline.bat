@echo off
setlocal

chdir /d D:\project
call mbti_env\Scripts\activate.bat

set PER_LABEL_LIMIT=%PIPELINE_PER_LABEL_LIMIT%
if "%PER_LABEL_LIMIT%"=="" set PER_LABEL_LIMIT=25

set TOTAL_LIMIT=%PIPELINE_TOTAL_LIMIT%
if "%TOTAL_LIMIT%"=="" set TOTAL_LIMIT=400

set AUDIO_DURATION=%PIPELINE_AUDIO_DURATION%
if "%AUDIO_DURATION%"=="" set AUDIO_DURATION=20

set MIN_AUDIO_DURATION=%PIPELINE_MIN_AUDIO_DURATION%
if "%MIN_AUDIO_DURATION%"=="" set MIN_AUDIO_DURATION=12

set MIN_AUDIO_SIZE_BYTES=%PIPELINE_MIN_AUDIO_SIZE_BYTES%
if "%MIN_AUDIO_SIZE_BYTES%"=="" set MIN_AUDIO_SIZE_BYTES=180000

set MIN_TRAIN_SAMPLES=%PIPELINE_MIN_TRAIN_SAMPLES%
if "%MIN_TRAIN_SAMPLES%"=="" set MIN_TRAIN_SAMPLES=64

set MIN_LABEL_COVERAGE=%PIPELINE_MIN_LABEL_COVERAGE%
if "%MIN_LABEL_COVERAGE%"=="" set MIN_LABEL_COVERAGE=8

echo ============================================
echo AUDIO2MBTI FULL CNN PIPELINE
echo ============================================
echo per_label_limit=%PER_LABEL_LIMIT%
echo total_limit=%TOTAL_LIMIT%
echo audio_duration=%AUDIO_DURATION%
echo min_audio_duration=%MIN_AUDIO_DURATION%
echo min_audio_size_bytes=%MIN_AUDIO_SIZE_BYTES%
echo min_train_samples=%MIN_TRAIN_SAMPLES%
echo min_label_coverage=%MIN_LABEL_COVERAGE%
echo.

python scripts\run_full_cnn_pipeline.py ^
  --per-label-limit %PER_LABEL_LIMIT% ^
  --total-limit %TOTAL_LIMIT% ^
  --duration %AUDIO_DURATION% ^
  --min-audio-duration %MIN_AUDIO_DURATION% ^
  --min-audio-size-bytes %MIN_AUDIO_SIZE_BYTES% ^
  --min-train-samples %MIN_TRAIN_SAMPLES% ^
  --min-label-coverage %MIN_LABEL_COVERAGE%

if errorlevel 1 (
  echo.
  echo ERROR: full CNN pipeline failed
  exit /b 1
)

echo.
echo FULL CNN PIPELINE COMPLETE
exit /b 0
