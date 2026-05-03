@echo off
echo Starting CampusVoice Backend API...
cd /d "%~dp0student-complaint-api"
call venv311\Scripts\activate.bat
python app.py
pause
