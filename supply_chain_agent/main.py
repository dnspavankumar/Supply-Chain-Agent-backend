from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import warnings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from supply_chain_agent.config import get_settings

# Suppress noisy third-party Pydantic warnings emitted by agno/google-genai models.
warnings.filterwarnings(
    "ignore",
    message=r'Field "model_(id|provider)" has conflict with protected namespace "model_"\.',
    category=UserWarning,
    module=r"pydantic\._internal\._fields",
)
warnings.filterwarnings(
    "ignore",
    message=r'Field name "(name|metadata|done|error)" shadows an attribute in parent "Operation";\s*',
    category=UserWarning,
    module=r"pydantic\._internal\._fields",
)

from supply_chain_agent.routers.agents import router as agents_router
from supply_chain_agent.routers.compliance import router as compliance_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.started_at = datetime.now(timezone.utc)
    app.state.settings = settings
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(compliance_router)
app.include_router(agents_router)
