from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import asyncio
from app.alerts.routes import ws_router
from app.audit.routes import router as audit_router
from app.audit.models import create_db_and_tables
from app.demo import router as demo_router
from app.admin.routes import router as admin_router
from app.test_bench.routes import router as test_bench_router
from app.live_mic.routes import router as live_mic_router
from fastapi.middleware.cors import CORSMiddleware
import traceback

app = FastAPI(title="TrueTone API", description="AI-Powered Real-Time Voice-Clone Detection")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal Server Error",
            "detail": str(exc)
        }
    )

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

app.include_router(ws_router)
app.include_router(audit_router)
app.include_router(demo_router)
app.include_router(admin_router)
app.include_router(test_bench_router)
app.include_router(live_mic_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "truetone-backend"}
