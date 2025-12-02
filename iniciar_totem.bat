@echo off
REM Movernos a la carpeta donde está este .bat
cd /d "%~dp0"

REM ------- PRIMERA TERMINAL: UI (python -m core.ui) -------
start "UI Totem" powershell.exe -NoExit -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process; .\.venv\Scripts\activate; python -m core.ui"

REM ------- SEGUNDA TERMINAL: API (uvicorn amain:app ...) -------
start "API Totem" powershell.exe -NoExit -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process; .\.venv\Scripts\activate; uvicorn amain:app --host 127.0.0.1 --port 8000 --reload"
