@echo off
echo Mandelbrot Renderer
echo ==================
echo.

REM Run with default as PNG if no arguments
if "%1"=="" (
    mandelbrot.exe -o mandelbrot.png
) else (
    mandelbrot.exe %*
)

pause