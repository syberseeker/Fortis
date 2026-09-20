import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .frameworks import seed_all_frameworks, FRAMEWORK_NAMES
from engine import check_model_available
from .store import init_db
from .routers import upload, chat, report, engagements

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fortis — AI Cybersecurity Advisor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(engagements.router)
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(report.router)


@app.on_event("startup")
async def startup():
    logger.info("Initializing client/engagement database")
    init_db()
    logger.info("Seeding framework reference corpus: %s", FRAMEWORK_NAMES)
    seed_all_frameworks()
    warning = await check_model_available()
    if warning:
        logger.warning(warning)


@app.get("/health")
async def health():
    model_warning = await check_model_available()
    return {
        "status": "ok" if not model_warning else "degraded",
        "detail": model_warning,
        "frameworks_loaded": FRAMEWORK_NAMES,
    }
