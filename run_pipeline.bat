@echo off
REM ============================================
REM Audio2MBTI - Pipeline Automation
REM Active flow:
REM - Kaggle playlist crawl/reprocess
REM - Recent release harvest from Spotify/YouTube/Apple Music seeds
REM - quality check
REM ============================================

chdir /d d:\project
call mbti_env\Scripts\activate.bat

echo.
echo ============================================
echo AUDIO2MBTI PIPELINE
echo ============================================
echo.

REM Phase 1: Data Collection (only with real data)

echo [1/3] Processing Kaggle Dataset...
python crawl\kaggle_mbti_reprocessor.py
if errorlevel 1 (
    echo.
    echo ERROR: Kaggle processing failed
    goto end
)

echo.
echo [2/3] Harvesting Recent Releases...
python crawl\recent_release_harvester.py --manifest config\recent_source_seeds.json --min-year 2024 --max-year 2026
if errorlevel 1 (
    echo.
    echo ERROR: Recent release harvest failed
    goto end
)

echo.
echo [3/3] Quality Check...
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
powershell -NoProfile -Command "Get-Content 'logs/audio2mbti.log' -Tail 20"
goto end

:end
pause
