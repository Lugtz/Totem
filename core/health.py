# core/health.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok", "message": "API working fine."})
