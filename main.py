from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from config import settings
import os
import logging
from routers import predict, retrain

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLOv8 Inference and Retraining API")

# --- Exception Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"An unexpected error occurred: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred"},
    )

# --- Routers ---
app.include_router(predict.router)
app.include_router(retrain.router)

# --- Startup Operations ---
@app.on_event("startup")
async def startup_event():
    """
    Create directories on startup.
    """
    os.makedirs(settings.INFERENCE_IMAGES_PATH, exist_ok=True)
    os.makedirs(settings.TEMP_ZIP_PATH, exist_ok=True)
    logger.info("Directories created successfully")

# --- Root Endpoint ---
@app.get("/")
def read_root():
    """
    Root endpoint to check if the API is running.
    """
    return {"status": "API is running"}
