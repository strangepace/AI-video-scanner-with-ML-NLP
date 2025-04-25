from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
import uuid
from typing import Optional, List
import logging
import base64
from pydantic import BaseModel
import yaml

# Import our custom modules
from advanced_detector import AdvancedVideoDetector
from utils import format_timestamp, get_video_info, extract_frame_preview

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Load configuration
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize detector
video_detector = AdvancedVideoDetector()

app = FastAPI(title="AI Video Search Tool")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body models
class SearchRequest(BaseModel):
    file_id: str
    prompt: str
    mode: Optional[str] = "full_scan"  # Default to full_scan

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file for processing
    """
    try:
        logger.info(f"Received file upload request: {file.filename}")
        
        if not file.filename.endswith(('.mp4', '.MP4')):
            raise HTTPException(status_code=400, detail="Only MP4 files are supported")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        file_path = UPLOAD_DIR / f"{file_id}{file_extension}"
        
        # Save the uploaded file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved successfully: {file_path}")
        
        return JSONResponse({
            "message": "File uploaded successfully",
            "file_id": file_id
        })
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_video(search_request: SearchRequest):
    """
    Search for objects/people in the video based on the prompt using the specified mode
    """
    try:
        file_id = search_request.file_id
        prompt = search_request.prompt
        mode = search_request.mode
        
        logger.info(f"Received search request - file_id: {file_id}, prompt: {prompt}, mode: {mode}")
        
        # Validate mode
        if mode not in ["full_scan", "frame_skip", "motion_filter", "track_then_match"]:
            raise HTTPException(status_code=400, detail="Invalid mode specified")
        
        # Check if mode is enabled in config
        if not config['modes'][mode]['enabled']:
            raise HTTPException(status_code=400, detail=f"Mode {mode} is not enabled")
        
        # Check if file exists
        file_path = None
        for ext in ['.mp4', '.MP4']:
            temp_path = UPLOAD_DIR / f"{file_id}{ext}"
            if temp_path.exists():
                file_path = temp_path
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Process video with selected mode
        logger.info(f"Processing video with mode: {mode}")
        results = video_detector.process_video(str(file_path), prompt, mode)
        
        # Format results
        formatted_results = []
        for result in results:
            timestamp = result['timestamp']
            for detection in result['detections']:
                # Extract relevant region from the frame
                frame = video_detector.get_frame_at_timestamp(str(file_path), timestamp)
                x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
                region = frame[y1:y2, x1:x2]
                
                # Create a preview
                preview_bytes = extract_frame_preview(frame)
                preview_base64 = base64.b64encode(preview_bytes).decode('utf-8')
                
                formatted_results.append({
                    "timestamp": format_timestamp(timestamp),
                    "confidence": detection.get('confidence', 0),
                    "preview": preview_base64,
                    "object_class": detection.get('class', 'unknown'),
                    "track_id": detection.get('track_id'),
                    "duration": detection.get('duration', 0)
                })
        
        # Sort by confidence and limit results
        formatted_results.sort(key=lambda x: x["confidence"], reverse=True)
        formatted_results = formatted_results[:config['general']['max_results']]
        
        if not formatted_results:
            logger.info(f"No matches found for prompt: {prompt}")
            return JSONResponse({
                "message": "No matches found for the given prompt",
                "file_id": file_id,
                "prompt": prompt,
                "mode": mode,
                "results": []
            })
        
        logger.info(f"Found {len(formatted_results)} matches for prompt: {prompt}")
        return JSONResponse({
            "message": "Search completed",
            "file_id": file_id,
            "prompt": prompt,
            "mode": mode,
            "results": formatted_results
        })
    except Exception as e:
        logger.error(f"Error processing search request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

@app.get("/modes")
async def get_available_modes():
    """
    Get available processing modes and their configurations
    """
    return {
        "modes": {
            mode: {
                "enabled": config['modes'][mode]['enabled'],
                "description": config['modes'][mode]['description']
            }
            for mode in ["full_scan", "frame_skip", "motion_filter", "track_then_match"]
        }
    } 