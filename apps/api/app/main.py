from __future__ import annotations

from fastapi import FastAPI

from .api.routes import router
from .core.config import settings
from .db import SQLiteDatabase
from .services.asr_service import ASRService

app = FastAPI(title=settings.app_name)
app.include_router(router, prefix="/api/v1", tags=["audio-typewriter"])


database = SQLiteDatabase()
asr_service = ASRService()


@app.on_event("startup")
def on_startup() -> None:
    settings.ensure_directories()
    database.initialize()
    asr_service.preload_default_model_if_configured()
