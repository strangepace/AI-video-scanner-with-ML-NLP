from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
from pathlib import Path
import uuid
import logging
import tempfile
from typing import List, Dict, Any, Tuple
import torch
import cv2
import numpy as np
from pyngrok import ngrok
import uvicorn
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch.nn.functional as F
from datetime import datetime
import json
import gc

# Enhanced logging setup
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

# Create directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
PREVIEW_DIR = UPLOAD_DIR / "previews"
PREVIEW_DIR.mkdir(exist_ok=True)

class Config:
    SIMILARITY_THRESHOLD = 0.25
    MIN_SEGMENT_DURATION = 1.0  # seconds
    SAMPLE_RATE = 5  # process every Nth frame
    MIN_DETECTION_CONFIDENCE = 0.35
    MAX_PREVIEW_FRAMES = 30

config = Config()

class SimpleProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize YOLO
        self.yolo = YOLO('yolov8m.pt')
        self.yolo.conf = config.MIN_DETECTION_CONFIDENCE
        logger.info("Loaded YOLOv8 model")
        
        # Initialize CLIP
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        logger.info("Loaded CLIP model")

    def process_frame(self, frame: np.ndarray, prompt: str) -> List[Dict[str, Any]]:
        """Process a single frame"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Run YOLO detection
            results = self.yolo(frame_rgb, verbose=False)[0]
            detections = []
            
            # Process each detection
            crops = []
            boxes = []
            
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Ensure valid coordinates
                x1, x2 = max(0, x1), min(frame.shape[1], x2)
                y1, y2 = max(0, y1), min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    crop = pil_img.crop((x1, y1, x2, y2))
                    crops.append(crop)
                    boxes.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_name': self.yolo.model.names[cls]
                    })
            
            # Batch process crops with CLIP
            if crops:
                inputs = self.clip_processor(
                    text=[prompt] * len(crops),
                    images=crops,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.clip(**inputs)
                    similarities = F.softmax(outputs.logits_per_image, dim=1).cpu().numpy()[:, 0]
                
                # Add matching detections
                for box, sim in zip(boxes, similarities):
                    if sim > config.SIMILARITY_THRESHOLD:
                        box['similarity'] = float(sim)
                        detections.append(box)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return []
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def save_preview(self, frames: List[np.ndarray], output_path: Path) -> bool:
        """Save a preview video clip"""
        try:
            if not frames:
                return False
            
            height, width = frames[0].shape[:2]
            out = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                30,
                (width, height)
            )
            
            # Limit number of preview frames
            preview_frames = frames[:config.MAX_PREVIEW_FRAMES]
            
            for frame in preview_frames:
                out.write(frame)
            
            out.release()
            return True
            
        except Exception as e:
            logger.error(f"Error saving preview: {str(e)}")
            return False

    def process_video(self, video_path: str, prompt: str) -> List[Dict[str, Any]]:
        """Process video using simple approach"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0
            
            segments = []
            current_segment = None
            
            try:
                while cap.isOpened() and frame_idx < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every Nth frame
                    if frame_idx % config.SAMPLE_RATE == 0:
                        detections = self.process_frame(frame, prompt)
                        timestamp = frame_idx / fps
                        
                        if detections:
                            if current_segment is None:
                                current_segment = {
                                    'start_time': timestamp,
                                    'end_time': timestamp,
                                    'preview_frames': [frame],
                                    'max_confidence': max(d['confidence'] for d in detections),
                                    'max_similarity': max(d['similarity'] for d in detections)
                                }
                            else:
                                # Check if this frame belongs to current segment
                                if timestamp - current_segment['end_time'] <= config.MIN_SEGMENT_DURATION:
                                    current_segment['end_time'] = timestamp
                                    current_segment['preview_frames'].append(frame)
                                    current_segment['max_confidence'] = max(
                                        current_segment['max_confidence'],
                                        max(d['confidence'] for d in detections)
                                    )
                                    current_segment['max_similarity'] = max(
                                        current_segment['max_similarity'],
                                        max(d['similarity'] for d in detections)
                                    )
                                else:
                                    # Save current segment
                                    if current_segment['end_time'] - current_segment['start_time'] >= config.MIN_SEGMENT_DURATION:
                                        preview_path = PREVIEW_DIR / f"segment_{len(segments)}_{int(current_segment['start_time'])}_{int(current_segment['end_time'])}.mp4"
                                        if self.save_preview(current_segment['preview_frames'], preview_path):
                                            segment_data = {
                                                'start_time': float(current_segment['start_time']),
                                                'end_time': float(current_segment['end_time']),
                                                'confidence': float(current_segment['max_confidence']),
                                                'similarity': float(current_segment['max_similarity']),
                                                'video_preview': str(preview_path.relative_to(UPLOAD_DIR.parent))
                                            }
                                            segments.append(segment_data)
                                    
                                    # Start new segment
                                    current_segment = {
                                        'start_time': timestamp,
                                        'end_time': timestamp,
                                        'preview_frames': [frame],
                                        'max_confidence': max(d['confidence'] for d in detections),
                                        'max_similarity': max(d['similarity'] for d in detections)
                                    }
                        
                        elif current_segment is not None:
                            # End current segment if it meets minimum duration
                            if current_segment['end_time'] - current_segment['start_time'] >= config.MIN_SEGMENT_DURATION:
                                preview_path = PREVIEW_DIR / f"segment_{len(segments)}_{int(current_segment['start_time'])}_{int(current_segment['end_time'])}.mp4"
                                if self.save_preview(current_segment['preview_frames'], preview_path):
                                    segment_data = {
                                        'start_time': float(current_segment['start_time']),
                                        'end_time': float(current_segment['end_time']),
                                        'confidence': float(current_segment['max_confidence']),
                                        'similarity': float(current_segment['max_similarity']),
                                        'video_preview': str(preview_path.relative_to(UPLOAD_DIR.parent))
                                    }
                                    segments.append(segment_data)
                            
                            current_segment = None
                    
                    frame_idx += 1
                    
                    # Clear GPU memory periodically
                    if frame_idx % (config.SAMPLE_RATE * 10) == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                
                # Handle last segment
                if current_segment is not None and current_segment['end_time'] - current_segment['start_time'] >= config.MIN_SEGMENT_DURATION:
                    preview_path = PREVIEW_DIR / f"segment_{len(segments)}_{int(current_segment['start_time'])}_{int(current_segment['end_time'])}.mp4"
                    if self.save_preview(current_segment['preview_frames'], preview_path):
                        segment_data = {
                            'start_time': float(current_segment['start_time']),
                            'end_time': float(current_segment['end_time']),
                            'confidence': float(current_segment['max_confidence']),
                            'similarity': float(current_segment['max_similarity']),
                            'video_preview': str(preview_path.relative_to(UPLOAD_DIR.parent))
                        }
                        segments.append(segment_data)
                
                return segments
                
            finally:
                cap.release()
                
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

# Initialize FastAPI app
app = FastAPI(title="AI Video Search Tool")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processor
processor = SimpleProcessor()

# Serve static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
async def root():
    return {"message": "AI Video Search API is running"}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file upload request: {file.filename}")
        
        if not file.filename.endswith(('.mp4', '.MP4')):
            raise HTTPException(status_code=400, detail="Only MP4 files are supported")
        
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        file_path = UPLOAD_DIR / f"{file_id}{file_extension}"
        
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
async def search_video(file_id: str, prompt: str):
    try:
        logger.info(f"Received search request - file_id: {file_id}, prompt: {prompt}")
        
        # Find video file
        file_path = None
        for ext in ['.mp4', '.MP4']:
            temp_path = UPLOAD_DIR / f"{file_id}{ext}"
            if temp_path.exists():
                file_path = temp_path
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Process video
        segments = processor.process_video(str(file_path), prompt)
        
        return JSONResponse({
            "message": "Search completed successfully",
            "results": segments
        })
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def start_server():
    try:
        # Start ngrok tunnel
        ngrok_tunnel = ngrok.connect(8000)
        logger.info(f"Public URL: {ngrok_tunnel.public_url}")
        
        # Start FastAPI server
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise

if __name__ == "__main__":
    start_server() 