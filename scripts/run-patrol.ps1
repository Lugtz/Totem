# Activa el venv .venv si existe, si no .venv311
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
if (Test-Path "$root\.venv\Scripts\Activate.ps1") {
    & "$root\.venv\Scripts\Activate.ps1"
} elseif (Test-Path "$root\.venv311\Scripts\Activate.ps1") {
    & "$root\.venv311\Scripts\Activate.ps1"
} else {
    throw "No encontré .venv ni .venv311 en $root"
}

# Ejecuta el patrullaje
python "$PSScriptRoot\yolo_person.py" --source 0 --width 3840 --height 2160 --out-w 1920 --out-h 1080
