from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from pathlib import Path
from utils.video_processor import VideoProcessor
from typing import Optional

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize video processor
video_processor = VideoProcessor()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Video Search API is running"}

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Validate video file
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Save the video file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {"message": "Video uploaded successfully", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-video")
async def search_video(filename: str, query: str):
    try:
        video_path = UPLOAD_DIR / filename
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Process video and search for objects
        results = video_processor.process_video(str(video_path), query)
        
        # Extract frames for each detection
        for result in results:
            frame_path = video_processor.get_frame(str(video_path), result["timestamp"])
            result["frame_path"] = frame_path
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video-frame")
async def get_video_frame(filename: str, timestamp: float):
    try:
        video_path = UPLOAD_DIR / filename
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")
        
        frame_path = video_processor.get_frame(str(video_path), timestamp)
        return {"frame_path": frame_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 