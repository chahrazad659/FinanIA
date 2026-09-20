@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 goto :fail
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto :fail
exit /b 0
:fail
echo Echec. Verifiez Python 3.10+ et la connexion pour installer les dependances.
pause
exit /b 1
