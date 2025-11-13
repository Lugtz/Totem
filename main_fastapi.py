# main_fastapi.py — Backend FastAPI Totem para /proposal/trigger
from fastapi import FastAPI
from core import proposal_trigger

api = FastAPI(title="Totem IA3 — Evolución i3")

# Montamos el router principal (Propuestas)
api.include_router(proposal_trigger.router)

@api.get("/health")
def health():
    return {"status": "ok"}
