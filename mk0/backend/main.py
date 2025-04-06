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

# Import our custom modules
from detector import VideoDetector
from clip_matcher import CLIPMatcher
from utils import format_timestamp, get_video_info, extract_frame_preview

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize models
video_detector = VideoDetector()
clip_matcher = CLIPMatcher()

app = FastAPI(title="AI Video Search Tool")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body model for search
class SearchRequest(BaseModel):
    file_id: str
    prompt: str

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
    Search for objects/people in the video based on the prompt
    """
    try:
        file_id = search_request.file_id
        prompt = search_request.prompt
        
        logger.info(f"Received search request - file_id: {file_id}, prompt: {prompt}")
        
        # Check if file exists
        file_path = None
        for ext in ['.mp4', '.MP4']:
            temp_path = UPLOAD_DIR / f"{file_id}{ext}"
            if temp_path.exists():
                file_path = temp_path
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Get video info
        video_info = get_video_info(str(file_path))
        logger.info(f"Video info: {video_info}")
        
        # Use a more aggressive sampling rate for faster processing
        # Calculate based on video duration - shorter videos can use smaller sample rates
        duration = video_info['duration']
        # For short videos (<30s), sample every 5 frames
        # For longer videos, sample proportionally to ensure ~100 total samples
        if duration < 30:
            sample_rate = 5
        else:
            # Aim for ~100 samples throughout the video
            estimated_frames = video_info['fps'] * duration
            sample_rate = max(5, int(estimated_frames / 100))
        
        logger.info(f"Using sample rate: {sample_rate} for video of duration {duration}s")
        
        # Process video with YOLO detector
        logger.info(f"Processing video with YOLO detector: {file_path}")
        frames_with_detections = video_detector.process_video(file_path, sample_rate=sample_rate)
        
        # Faster CLIP matching with lower threshold to catch more potential matches
        logger.info(f"Filtering results with CLIP matcher using prompt: {prompt}")
        
        results = []
        similarity_threshold = 0.2  # Lower threshold to increase recall
        
        for frame_data in frames_with_detections:
            timestamp = frame_data["timestamp"]
            frame = video_detector.get_frame_at_timestamp(file_path, timestamp)
            
            # Match detections with the prompt
            matched_detections = clip_matcher.match_detections(
                frame, 
                frame_data["detections"], 
                prompt,
                similarity_threshold=similarity_threshold
            )
            
            if matched_detections:
                # Get the best match (highest similarity)
                best_match = max(matched_detections, key=lambda x: x.get('similarity', 0))
                
                # Extract relevant region from the frame
                x1, y1, x2, y2 = [int(coord) for coord in best_match['bbox']]
                region = frame[y1:y2, x1:x2]
                
                # Create a preview
                preview_bytes = extract_frame_preview(frame)
                preview_base64 = base64.b64encode(preview_bytes).decode('utf-8')
                
                # Format the timestamp
                formatted_time = format_timestamp(timestamp)
                
                results.append({
                    "timestamp": formatted_time,
                    "confidence": best_match.get('similarity', 0),
                    "preview": preview_base64,
                    "object_class": best_match.get('class', 'unknown')
                })
        
        # Limit results to top 10 matches instead of just 5
        results = sorted(results, key=lambda x: x["confidence"], reverse=True)[:10]
        
        if not results:
            logger.info(f"No matches found for prompt: {prompt}")
            return JSONResponse({
                "message": "No matches found for the given prompt",
                "file_id": file_id,
                "prompt": prompt,
                "results": []
            })
        
        logger.info(f"Found {len(results)} matches for prompt: {prompt}")
        return JSONResponse({
            "message": "Search completed",
            "file_id": file_id,
            "prompt": prompt,
            "results": results
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