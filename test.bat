@echo off
:: Change directory to the script's location
cd /d "%~dp0"

echo ==============================================
echo   Starting Information Group Dashboard...
echo   Please wait, the browser will open shortly.
echo ==============================================

:: Run the dashboard
python -m streamlit run index.py

:: If the program crashes, pause here so you can read the error
echo.
echo ----------------------------------------------
echo   Program stopped. Press any key to close.
echo ----------------------------------------------
pause