@echo off
REM ============================================
REM Audio2MBTI - Pipeline Automation
REM Runs only crawlers with actual data:
REM - kaggle_mbti_reprocessor
REM - farm_modern_playlists
REM Then: aggregate + quality check
REM ============================================

chdir /d d:\project
call mbti_env\Scripts\activate.bat

echo.
echo ============================================
echo AUDIO2MBTI PIPELINE
echo ============================================
echo.

REM Phase 1: Data Collection (only with real data)

echo [1/4] Processing Kaggle Dataset...
python crawl\kaggle_mbti_reprocessor.py
if errorlevel 1 (
    echo.
    echo ERROR: Kaggle processing failed
    goto end
)

echo.
echo [2/4] Processing Modern Playlists...
python crawl\farm_modern_playlists.py
if errorlevel 1 (
    echo.
    echo ERROR: Modern playlists failed
    goto end
)

REM Phase 2: Data Processing

echo.
echo [3/4] Aggregating Data...
python crawl\aggregate_training_data.py
if errorlevel 1 (
    echo.
    echo ERROR: Aggregation failed
    goto end
)

echo.
echo [4/4] Quality Check...
python crawl\check_data_quality.py
if errorlevel 1 (
    echo.
    echo ERROR: Quality check failed
    goto end
)

echo.
echo ============================================
echo PIPELINE COMPLETE!
echo ============================================
echo.
echo Output file: data\mbti_master_training_data.csv
echo Logs: logs\audio2mbti.log
echo.
echo Last 20 lines of log:
type logs\audio2mbti.log | findstr /V "^$" | tail -n 20
goto end

:end
pause
