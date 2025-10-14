"""
Vercel-compatible API for NAVIQ
Lightweight FastAPI service for Vercel serverless deployment
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from datetime import datetime

# Create FastAPI app
app = FastAPI(
    title="NAVIQ API",
    description="NAVIQ GNSS Error Prediction API Service",
    version="1.0.0"
)

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "NAVIQ",
        "status": "active",
        "message": "Navigation Intelligence & Quality Assurance",
        "description": "GNSS Error Prediction API Service",
        "endpoints": ["/health", "/info", "/docs"],
        "dashboard": "Full dashboard available on Railway deployment"
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "naviq-api",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/info")
def info():
    """Service information"""
    return {
        "name": "NAVIQ GNSS Error Prediction",
        "purpose": "AI/ML based GNSS satellite error forecasting",
        "features": [
            "7-day training to Day-8 prediction",
            "Clock and ephemeris error prediction", 
            "Statistical normality optimization",
            "Multi-orbit satellite support (GEO/GSO/MEO)"
        ],
        "tech_stack": "Python + FastAPI + PyTorch + Streamlit",
        "deployment": "Vercel (API) + Railway (Dashboard)"
    }

# Export the app for Vercel
handler = app
