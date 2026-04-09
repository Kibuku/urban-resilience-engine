@echo off
REM Urban Resilience Engine - Setup and Management Batch Script

setlocal enabledelayedexpansion

set PROJECT_DIR=%~dp0
cd /d %PROJECT_DIR%

:menu
echo.
echo ================================================================
echo  Urban Resilience Engine - Project Management
echo ================================================================
echo.
echo  1. Install/update dependencies
echo  2. Run infrastructure test
echo  3. Verify API access
echo  4. Setup Google Earth Engine
echo  5. Start Phase 1 ETL pipeline
echo  6. Launch Streamlit dashboard
echo  7. Start ngrok demo (public URL)
echo  8. Check Python version
echo  9. View project structure
echo 10. Open quick reference
echo  0. Exit
echo.
set /p choice="Select option (0-10): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto test
if "%choice%"=="3" goto verify_api
if "%choice%"=="4" goto setup_gee
if "%choice%"=="5" goto phase1
if "%choice%"=="6" goto streamlit
if "%choice%"=="7" goto ngrok_demo
if "%choice%"=="8" goto python_check
if "%choice%"=="9" goto structure
if "%choice%"=="10" goto quickref
if "%choice%"=="0" goto end
goto menu

:install
echo.
echo Installing/updating dependencies...
echo.
py -m pip install -r requirements-py314.txt
echo.
echo Installation complete!
pause
goto menu

:test
echo.
echo Running infrastructure test...
echo.
py test_infrastructure.py
pause
goto menu

:verify_api
echo.
echo Verifying API access...
echo.
py verify_api_access.py
pause
goto menu

:setup_gee
echo.
echo Setting up Google Earth Engine...
echo.
py setup_gee.py
pause
goto menu

:phase1
echo.
echo Starting Phase 1 ETL Pipeline...
echo.
python -m src.phase1_etl.pipeline
pause
goto menu

:streamlit
echo.
echo Launching Streamlit dashboard...
echo.
echo Dashboard will open at http://localhost:8501
echo.
streamlit run src/phase4_deploy/app.py
goto menu

:ngrok_demo
echo.
echo Starting ngrok demo...
echo.
echo This will create a public URL for your dashboard demo.
echo Make sure NGROK_AUTH_TOKEN is set in your .env file.
echo.
python demo_ngrok.py
pause
goto menu

:python_check
echo.
py --version
echo.
py -c "import sys; print(f'Version: {sys.version}')"
echo.
pause
goto menu

:structure
echo.
echo Project Structure:
echo.
tree /F
echo.
pause
goto menu

:quickref
echo.
echo Opening quick reference...
echo.
if exist QUICK_START.md (
    start QUICK_START.md
) else (
    echo QUICK_START.md not found!
)
pause
goto menu

:end
echo.
echo Goodbye!
echo.
