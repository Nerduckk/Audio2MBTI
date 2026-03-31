@echo off
setlocal

cd /d D:\project
if not exist logs mkdir logs
if not exist outputs mkdir outputs

set "LOG_FILE=D:\project\logs\cnn_train.log"
set "PID_FILE=D:\project\outputs\cnn_train.pid"
set "STOP_FILE=D:\project\outputs\stop_full_cnn_pipeline.txt"
set "PYTHON_EXE=D:\project\mbti_env\Scripts\python.exe"
set "SCRIPT_PATH=D:\project\scripts\run_full_cnn_pipeline.py"

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=toggle"

if /I "%ACTION%"=="toggle" goto toggle
if /I "%ACTION%"=="start" goto start
if /I "%ACTION%"=="stop" goto stop
if /I "%ACTION%"=="kill" goto kill
if /I "%ACTION%"=="status" goto status
if /I "%ACTION%"=="run" goto run

echo usage: cnn_train.bat [start^|stop^|kill^|status]
exit /b 1

:toggle
call :resolve_pid
if defined FOUND_PID goto stop
goto start

:start
call :resolve_pid
if defined FOUND_PID (
  echo cnn trainer is already running. pid: %FOUND_PID%
  echo log: %LOG_FILE%
  exit /b 0
)
if exist "%STOP_FILE%" del /f /q "%STOP_FILE%" >nul 2>&1
echo starting cnn trainer in background...
start "cnn-train" /min cmd /c ""%PYTHON_EXE%" "%SCRIPT_PATH%" --continuous --skip-crawl --per-label-limit 0 --total-limit 0 --duration 20 --min-audio-duration 12 --min-audio-size-bytes 180000 --min-train-samples 64 --min-label-coverage 8 --loop-sleep-seconds 30 --stop-file "%STOP_FILE%" >> "%LOG_FILE%" 2>&1"
timeout /t 2 >nul
call :resolve_pid
if not defined FOUND_PID (
  echo failed to start cnn trainer. check log: %LOG_FILE%
  exit /b 1
)
> "%PID_FILE%" echo %FOUND_PID%
echo started. pid: %FOUND_PID%
echo pid file: %PID_FILE%
echo log: %LOG_FILE%
echo stop gracefully: cnn_train.bat stop
echo stop immediately: cnn_train.bat kill
exit /b 0

:stop
> "%STOP_FILE%" echo stop
call :resolve_pid
echo stop requested. current run will finish, then the loop will exit.
echo stop file: %STOP_FILE%
if defined FOUND_PID echo current pid: %FOUND_PID%
exit /b 0

:kill
powershell -NoProfile -Command "$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_full_cnn_pipeline.py*' -and $_.CommandLine -like '*--continuous*' -and $_.CommandLine -like '*--skip-crawl*' }; if (-not $procs) { exit 1 }; $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; Write-Output $_.ProcessId }" > "%PID_FILE%.killed.tmp"
if errorlevel 1 (
  echo cnn trainer is not running.
  exit /b 0
)
set "FOUND_PID="
for /f %%I in (%PID_FILE%.killed.tmp) do (
  if not defined FOUND_PID set "FOUND_PID=%%I"
)
del /f /q "%PID_FILE%.killed.tmp" >nul 2>&1
del /f /q "%PID_FILE%" >nul 2>&1
if exist "%STOP_FILE%" del /f /q "%STOP_FILE%" >nul 2>&1
echo killed cnn trainer process(es).
if defined FOUND_PID echo first pid: %FOUND_PID%
exit /b 0

:status
call :resolve_pid
if not defined FOUND_PID (
  echo cnn trainer is not running.
  echo log: %LOG_FILE%
  if exist "%PID_FILE%" echo stale pid file: %PID_FILE%
  exit /b 0
)
echo cnn trainer is running.
echo pid: %FOUND_PID%
echo log: %LOG_FILE%
if exist "%STOP_FILE%" echo graceful stop has been requested.
exit /b 0

:run
"%PYTHON_EXE%" "%SCRIPT_PATH%" --continuous --skip-crawl --per-label-limit 0 --total-limit 0 --duration 20 --min-audio-duration 12 --min-audio-size-bytes 180000 --min-train-samples 64 --min-label-coverage 8 --loop-sleep-seconds 30 --stop-file "%STOP_FILE%" >> "%LOG_FILE%" 2>&1
exit /b %ERRORLEVEL%

:resolve_pid
set "FOUND_PID="
for /f %%I in ('powershell -NoProfile -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_full_cnn_pipeline.py*' -and $_.CommandLine -like '*--continuous*' -and $_.CommandLine -like '*--skip-crawl*' } | Sort-Object CreationDate -Descending | Select-Object -First 1 -ExpandProperty ProcessId; if ($p) { $p }"') do set "FOUND_PID=%%I"
goto :eof
