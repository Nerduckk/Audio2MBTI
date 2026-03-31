@echo off
setlocal

cd /d D:\project
set "PYTHON_EXE=D:\project\mbti_env\Scripts\python.exe"
set "WORKFLOW_SCRIPT=D:\project\scripts\project_workflow.py"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
echo Running unified audio crawl workflow...
"%PYTHON_EXE%" "%WORKFLOW_SCRIPT%" crawl-audio

set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Crawl finished with exit code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
