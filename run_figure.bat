@echo off
REM MARS Figure Generation Script
REM This script installs dependencies and generates the study area figure

echo ============================================================
echo MARS Study Area Figure Generator - Setup and Run
echo ============================================================
echo.

REM Check if running in conda environment
where conda >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Conda detected. Using conda to install cartopy...
    echo Installing packages with conda...
    conda install -c conda-forge matplotlib cartopy pandas -y
    if %ERRORLEVEL% NEQ 0 (
        echo WARNING: Conda installation failed. Trying pip...
        pip install -r requirements_figure.txt
    )
) else (
    echo Conda not found. Installing with pip...
    pip install -r requirements_figure.txt
)

echo.
echo ============================================================
echo Running figure generation script...
echo ============================================================
python figure.py

echo.
echo ============================================================
echo Done!
echo ============================================================
pause
