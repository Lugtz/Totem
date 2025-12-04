@echo off
REM --- Detener procesos del Totem API ---

echo.
echo ==========================================================
echo 🛑 INTENTANDO DETENER SERVICIOS
echo (Esto cerrará TODAS las instancias de estos programas)
echo ==========================================================
echo.

REM Detener TODAS las instancias de Python
taskkill /f /im python.exe
echo [OK] Intentando detener python.exe

REM Detener TODAS las instancias de Uvicorn (si corre en un proceso separado)
REM NOTA: Si uvicorn corre dentro de python.exe, este comando puede no ser necesario
taskkill /f /im uvicorn
echo [OK] Intentando detener uvicorn

REM Detener TODAS las instancias de ngrok
taskkill /f /im ngrok.exe
echo [OK] Intentando detener ngrok.exe

echo.
echo ==========================================================
echo ✅ PROCESO DE TERMINACIÓN FINALIZADO.
echo Revisa los mensajes anteriores para ver si hubo errores.
echo ==========================================================
pause