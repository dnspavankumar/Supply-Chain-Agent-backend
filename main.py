from __future__ import annotations

import warnings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from config import get_settings
from services.errors import ComplianceViolationError

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

from routers.agents import router as agents_router
from routers.audit import router as audit_router
from routers.pipeline import router as pipeline_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
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


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
    }


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ComplianceViolationError)
async def compliance_violation_handler(
    _request: Request,
    exc: ComplianceViolationError,
) -> JSONResponse:
    return JSONResponse(status_code=422, content=exc.payload)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, _exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(agents_router)
app.include_router(audit_router)
app.include_router(pipeline_router)
