import io
from typing import Literal
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fashn_vton import TryOnPipeline
from PIL import Image
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the try-on pipeline on startup"""
    global pipeline
    logger.info("Initializing TryOnPipeline...")
    
    try:
        pipeline = TryOnPipeline(weights_dir="./models")
        logger.info("TryOnPipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise

    yield

    if pipeline:
        pipeline.unload()
        logger.info("TryOnPipeline unloaded successfully")

# Initialize FastAPI app
app = FastAPI(
    title="Fashion Try-On API",
    description="API for virtual fashion try-on using AI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Fashion Try-On API is running"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None
    }

@app.post("/try-on")
async def try_on(
    person_image: UploadFile = File(..., description="Person image file"),
    garment_image: UploadFile = File(..., description="Garment image file"),
    category: Literal["tops", "bottoms", "one-pieces"] = Form(
        ..., description="Garment category"
    )
):
    """
    Virtual try-on endpoint
    """

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Please try again later."
        )

    try:
        # Validate file types
        allowed_types = {
            "image/png",
            "image/jpeg",
            "image/jpg",
            "image/webp"
        }

        if person_image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid person image type: {person_image.content_type}"
            )

        if garment_image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid garment image type: {garment_image.content_type}"
            )

        logger.info(f"Processing try-on request with category: {category}")

        # Read uploaded files
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()

        # Convert to PIL images
        person = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment = Image.open(io.BytesIO(garment_bytes)).convert("RGB")

        logger.info(
            f"Person image size: {person.size}, "
            f"Garment image size: {garment.size}"
        )

        # Free unused GPU memory
        torch.cuda.empty_cache()

        # Resize images for lower VRAM usage
        person.thumbnail((384,384))
        garment.thumbnail((384, 384))

        # Run inference
        result = pipeline(
            person_image=person,
            garment_image=garment,
            category=category,
            
        )

        # Convert output image to bytes
        output_image = result.images[0]

        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        logger.info("Try-on processing completed successfully")

        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=tryon_result.png"
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            f"Error during try-on processing: {e}",
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process try-on: {str(e)}"
        )