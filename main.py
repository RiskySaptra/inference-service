from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from config import settings
import os
import logging
from routers import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLOv8 Inference API")

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

app.include_router(predict.router)

@app.on_event("startup")
async def startup_event():
    os.makedirs(settings.INFERENCE_IMAGES_PATH, exist_ok=True)
    logger.info("Directories created successfully")
    predict.load_model()

@app.get("/")
def read_root():
    return {"status": "API is running"}
