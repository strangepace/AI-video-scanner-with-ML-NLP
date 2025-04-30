import logging
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
import json
from typing import List, Dict, Any
import time

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "backend.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self):
        self.current_video = None
        self.video_frames = []
        self.frame_timestamps = []
        self.detected_segments = []
        
    def clear(self):
        self.current_video = None
        self.video_frames = []
        self.frame_timestamps = []
        self.detected_segments = []
        torch.cuda.empty_cache()
        logger.info("Memory cleared")

class VideoDetector:
    def __init__(self):
        logger.info("Initializing YOLO model...")
        self.model = YOLO('yolov8n.pt')
        logger.info("YOLO model initialized")
        
    def process_frame(self, frame):
        results = self.model(frame, verbose=False)[0]
        return results.boxes.data.cpu().numpy()

class CLIPMatcher:
    def __init__(self):
        logger.info("Initializing CLIP model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logger.info("CLIP model initialized")
        
    def compute_similarity(self, image, text):
        try:
            inputs = self.processor(
                images=image,
                text=[text],
                return_tensors="pt",
                padding=True
            )
            outputs = self.model(**inputs)
            return float(outputs.logits_per_image[0][0])
        except Exception as e:
            logger.error(f"Error computing CLIP similarity: {str(e)}")
            return 0.0

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory_manager = MemoryManager()
video_detector = VideoDetector()
clip_matcher = CLIPMatcher()

def save_segment_preview(frames: List[np.ndarray], segment_id: int) -> str:
    if not frames:
        logger.warning(f"No frames to save for segment {segment_id}")
        return ""
    
    preview_path = f"previews/segment_{segment_id}.jpg"
    os.makedirs("previews", exist_ok=True)
    
    try:
        middle_frame = frames[len(frames) // 2]
        cv2.imwrite(preview_path, middle_frame)
        logger.info(f"Saved preview for segment {segment_id}")
        return preview_path
    except Exception as e:
        logger.error(f"Error saving preview for segment {segment_id}: {str(e)}")
        return ""

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        logger.info(f"Receiving video upload: {file.filename}")
        memory_manager.clear()
        
        # Save video temporarily
        temp_path = f"temp_{int(time.time())}.mp4"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process video
        cap = cv2.VideoCapture(temp_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            memory_manager.video_frames.append(frame)
            memory_manager.frame_timestamps.append(frame_count / cap.get(cv2.CAP_PROP_FPS))
            frame_count += 1
            
        cap.release()
        os.remove(temp_path)
        
        logger.info(f"Video processed: {frame_count} frames")
        return {"message": "Video uploaded successfully", "frames": frame_count}
        
    except Exception as e:
        logger.error(f"Error processing video upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_video(search_request: Dict[str, Any]):
    try:
        query = search_request.get("query", "")
        logger.info(f"Processing search request: {query}")
        
        if not memory_manager.video_frames:
            raise HTTPException(status_code=400, detail="No video loaded")
            
        segments = []
        current_segment = None
        
        for i, frame in enumerate(memory_manager.video_frames):
            detections = video_detector.process_frame(frame)
            
            for det in detections:
                x1, y1, x2, y2, conf, cls = det[:6]
                if conf < 0.5:
                    continue
                    
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue
                    
                similarity = clip_matcher.compute_similarity(crop, query)
                
                if similarity > 0.25:  # Threshold for matching
                    timestamp = memory_manager.frame_timestamps[i]
                    
                    if current_segment is None:
                        current_segment = {
                            "start_time": timestamp,
                            "end_time": timestamp,
                            "frames": [frame],
                            "confidence": float(conf),
                            "similarity": float(similarity)
                        }
                    else:
                        current_segment["end_time"] = timestamp
                        current_segment["frames"].append(frame)
                        current_segment["confidence"] = max(current_segment["confidence"], float(conf))
                        current_segment["similarity"] = max(current_segment["similarity"], float(similarity))
                        
                elif current_segment is not None:
                    preview_path = save_segment_preview(current_segment["frames"], len(segments))
                    segments.append({
                        "start_time": current_segment["start_time"],
                        "end_time": current_segment["end_time"],
                        "preview_path": preview_path,
                        "confidence": current_segment["confidence"],
                        "similarity": current_segment["similarity"]
                    })
                    current_segment = None
        
        # Handle last segment
        if current_segment is not None:
            preview_path = save_segment_preview(current_segment["frames"], len(segments))
            segments.append({
                "start_time": current_segment["start_time"],
                "end_time": current_segment["end_time"],
                "preview_path": preview_path,
                "confidence": current_segment["confidence"],
                "similarity": current_segment["similarity"]
            })
        
        # Sort segments by start time
        segments.sort(key=lambda x: x["start_time"])
        
        logger.info(f"Search completed. Found {len(segments)} segments")
        return {"segments": segments}
        
    except Exception as e:
        logger.error(f"Error processing search request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting backend server...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 