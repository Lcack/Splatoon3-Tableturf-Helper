@echo off
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"
start "" "%PROJECT_ROOT%\.venv\Scripts\tableturf-video-review.exe"
