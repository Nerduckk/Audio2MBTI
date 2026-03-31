@echo off
setlocal

cd /d D:\project
echo Legacy wrapper: forwarding to project root crawl_audio_data.bat
call D:\project\crawl_audio_data.bat
exit /b %ERRORLEVEL%
