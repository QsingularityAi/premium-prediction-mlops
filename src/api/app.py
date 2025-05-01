import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import uvicorn
import yaml
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

# Add src to Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.model_predictor import ModelPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.getenv("LOG_DIR", "logs"), "api.log")),
    ],
)
logger = logging.getLogger(__name__)


# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title="Premium Prediction API",
    description="API for premium prediction using segmented regression models",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
prediction_counter = Counter(
    "premium_predictions_total",
    "Total number of premium predictions made",
    ["status", "segment"],
)
prediction_latency = Histogram(
    "premium_prediction_latency_seconds",
    "Latency of premium predictions in seconds",
    ["segment"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
)
model_errors = Counter(
    "premium_model_errors_total", "Total number of model errors", ["error_type"]
)
active_requests = Gauge(
    "premium_active_requests", "Number of active prediction requests"
)


# Request and response models
class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Features for prediction")


class BatchPredictionRequest(BaseModel):
    instances: List[Dict[str, Any]] = Field(
        ..., description="List of feature instances for batch prediction"
    )


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted premium amount")
    segment: str = Field(..., description="Premium segment")
    success: bool = Field(..., description="Success flag")
    message: str = Field(..., description="Status message")
    request_id: str = Field(..., description="Unique request ID")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    predictions: List[float] = Field(
        ..., description="List of predicted premium amounts"
    )
    segments: List[str] = Field(..., description="List of premium segments")
    success: bool = Field(..., description="Success flag")
    message: str = Field(..., description="Status message")
    request_id: str = Field(..., description="Unique request ID")
    timestamp: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_version: str = Field(..., description="Model version")
    timestamp: str = Field(..., description="Current timestamp")


# Initialize model predictor
model_predictor = ModelPredictor(config)


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("Starting up Premium Prediction API")
    # Load the models
    version = os.getenv("MODEL_VERSION", "latest")
    success = model_predictor.load_models(version)
    if not success:
        logger.error(f"Failed to load models version {version}")
        # We'll continue and let health checks fail until models are loaded


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if not model_predictor.segment_models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    return {
        "status": "healthy",
        "version": "1.0.0",
        "model_version": os.getenv("MODEL_VERSION", "latest"),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/ready", response_model=HealthResponse, tags=["Health"])
async def readiness_check():
    """Readiness check endpoint"""
    if not model_predictor.segment_models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    return {
        "status": "ready",
        "version": "1.0.0",
        "model_version": os.getenv("MODEL_VERSION", "latest"),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Expose Prometheus metrics"""
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Make single premium prediction
    """
    active_requests.inc()
    request_id = f"req_{int(time.time() * 1000)}"
    start_time = time.time()

    try:
        # Convert feature dictionary to DataFrame
        features_df = pd.DataFrame([request.features])

        # Make prediction
        result = model_predictor.predict(features_df)

        if not result["success"]:
            model_errors.labels(error_type="prediction_error").inc()
            raise HTTPException(status_code=500, detail=result["message"])

        # Create response
        response = {
            "prediction": float(result["predictions"][0]),
            "segment": result["segments"][0],
            "success": True,
            "message": "Prediction successful",
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
        }

        # Record metrics
        latency = time.time() - start_time
        prediction_counter.labels(status="success", segment=result["segments"][0]).inc()
        prediction_latency.labels(segment=result["segments"][0]).observe(latency)

        # Log request asynchronously
        background_tasks.add_task(
            log_prediction_request,
            request_id=request_id,
            features=request.features,
            prediction=response["prediction"],
            segment=response["segment"],
        )

        active_requests.dec()
        return response

    except Exception as e:
        model_errors.labels(error_type="server_error").inc()
        prediction_counter.labels(status="error", segment="unknown").inc()
        active_requests.dec()
        logger.error(f"Error processing prediction request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(
    request: BatchPredictionRequest, background_tasks: BackgroundTasks
):
    """
    Make batch premium predictions
    """
    active_requests.inc()
    request_id = f"batch_{int(time.time() * 1000)}"
    start_time = time.time()

    try:
        # Convert instances to DataFrame
        features_df = pd.DataFrame(request.instances)

        # Make batch predictions
        result = model_predictor.predict_batch(features_df)

        if not result["success"]:
            model_errors.labels(error_type="batch_prediction_error").inc()
            raise HTTPException(status_code=500, detail=result["message"])

        # Create response
        response = {
            "predictions": [float(p) for p in result["predictions"]],
            "segments": result["segments"],
            "success": True,
            "message": "Batch prediction successful",
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
        }

        # Record metrics
        latency = time.time() - start_time
        for segment in set(result["segments"]):
            count = result["segments"].count(segment)
            prediction_counter.labels(status="success", segment=segment).inc(count)
            prediction_latency.labels(segment=segment).observe(
                latency / len(request.instances)
            )

        # Log batch prediction asynchronously
        background_tasks.add_task(
            log_batch_prediction,
            request_id=request_id,
            count=len(request.instances),
            predictions=response["predictions"],
            segments=response["segments"],
        )

        active_requests.dec()
        return response

    except Exception as e:
        model_errors.labels(error_type="server_error").inc()
        prediction_counter.labels(status="error", segment="unknown").inc()
        active_requests.dec()
        logger.error(f"Error processing batch prediction request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


def log_prediction_request(
    request_id: str, features: Dict, prediction: float, segment: str
):
    """Log prediction request asynchronously"""
    try:
        log_data = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "features": features,
            "prediction": prediction,
            "segment": segment,
        }

        # Log to file in a JSON format
        log_dir = os.getenv("LOG_DIR", "logs")
        log_path = os.path.join(log_dir, "predictions.jsonl")
        os.makedirs(log_dir, exist_ok=True)

        with open(log_path, "a") as f:
            f.write(json.dumps(log_data) + "\n")

    except Exception as e:
        logger.error(f"Error logging prediction: {str(e)}")


def log_batch_prediction(
    request_id: str, count: int, predictions: List[float], segments: List[str]
):
    """Log batch prediction request asynchronously"""
    try:
        log_data = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "count": count,
            "prediction_stats": {
                "min": float(min(predictions)),
                "max": float(max(predictions)),
                "mean": float(np.mean(predictions)),
                "median": float(np.median(predictions)),
            },
            "segment_counts": {
                segment: segments.count(segment) for segment in set(segments)
            },
        }

        # Log to file in a JSON format
        log_dir = os.getenv("LOG_DIR", "logs")
        log_path = os.path.join(log_dir, "batch_predictions.jsonl")
        os.makedirs(log_dir, exist_ok=True)

        with open(log_path, "a") as f:
            f.write(json.dumps(log_data) + "\n")

    except Exception as e:
        logger.error(f"Error logging batch prediction: {str(e)}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": str(exc.detail),
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": f"Internal server error: {str(exc)}",
            "timestamp": datetime.now().isoformat(),
        },
    )


if __name__ == "__main__":
    # Run the API server directly if this file is executed
    port = int(os.getenv("PORT", config["api"]["port"]))
    log_level = os.getenv("LOG_LEVEL", config["api"]["log_level"])

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level=log_level.lower(),
        workers=config["api"]["workers"],
        reload=True,
    )
