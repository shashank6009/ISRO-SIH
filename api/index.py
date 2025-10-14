"""
Vercel-compatible FastAPI wrapper for NAVIQ
This serves as the main entrypoint for Vercel deployment
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
import os

app = FastAPI(
    title="NAVIQ Inference Service",
    description="Navigation Intelligence & Quality Assurance - GNSS Error Prediction API",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint - redirects to dashboard info"""
    return {
        "service": "naviq",
        "status": "active",
        "message": "NAVIQ GNSS Error Prediction System",
        "dashboard_note": "This is the API service. The main dashboard is a Streamlit app that should be deployed separately.",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "naviq",
        "timestamp": "2024-10-14T21:04:31Z"
    }

@app.get("/predict")
async def predict_placeholder():
    """Placeholder prediction endpoint"""
    return {
        "message": "GNSS Error Prediction API",
        "note": "Full prediction functionality requires the complete NAVIQ system deployment",
        "status": "service_available"
    }

# Import the actual FastAPI service if available
try:
    from src.genesis_ai.inference.service import app as inference_app
    # Mount the inference service
    app.mount("/api", inference_app)
except ImportError:
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
