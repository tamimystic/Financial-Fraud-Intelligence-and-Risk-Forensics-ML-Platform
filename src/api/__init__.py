"""
FastAPI Microservice Sub-Package.
"""

from src.api.app import create_app

app = create_app()

__all__ = ["app", "create_app"]
