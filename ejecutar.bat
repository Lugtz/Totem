@echo off
REM ==== Usar la carpeta donde está este .bat como raíz del proyecto ====
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

REM ==== Lanzar backend FastAPI (Totem API) ====
start "Totem API" powershell -NoExit -Command ^
"Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned; ^
cd '%PROJECT_DIR%'; ^
.\.venv311\Scripts\Activate.ps1; ^
uvicorn amain:app --host 127.0.0.1 --port 8000 --reload"

REM ==== Lanzar UI 3D de Nacho ====
start "Totem UI" powershell -NoExit -Command ^
"Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned; ^
cd '%PROJECT_DIR%'; ^
.\.venv311\Scripts\Activate.ps1; ^
python -m core.ui"

REM ==== Lanzar servidor HTTP en puerto 9000 (carpeta infografias) ====
start "HTTP 9000" powershell -NoExit -Command ^
"Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned; ^
cd '%PROJECT_DIR%'; ^
.\.venv311\Scripts\Activate.ps1; ^
python -m http.server 9000 -d infografias"

REM ==== Lanzar ngrok para exponer el puerto 9000 ====
start "ngrok 9000" powershell -NoExit -Command ^
"Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned; ^
cd '%PROJECT_DIR%'; ^
ngrok http 9000"
