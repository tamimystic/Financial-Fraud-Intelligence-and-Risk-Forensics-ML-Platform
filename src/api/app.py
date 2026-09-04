"""
FastAPI Application Factory.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise Financial Fraud Intelligence & Risk Forensics Gateway",
        description="Sub-5ms Real-Time Fraud Scoring, 4-Tier Adaptive Routing, and FCRA Adverse Action API.",
        version="2.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.include_router(router, prefix="/api/v1")
    return app
