@echo off
setlocal

cd /d D:\project
set "PYTHON_EXE=D:\project\mbti_env\Scripts\python.exe"
set "WORKFLOW_SCRIPT=D:\project\scripts\project_workflow.py"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

echo Running hybrid refresh workflow...
"%PYTHON_EXE%" "%WORKFLOW_SCRIPT%" refresh-hybrid

set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Hybrid refresh finished with exit code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
