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
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

class Config:
    SIMILARITY_THRESHOLD = 0.25
    MIN_SEGMENT_DURATION = 1.0  # seconds
    SAMPLE_RATE_FIRST_PASS = 15  # process every Nth frame in first pass
    SAMPLE_RATE_SECOND_PASS = 5  # process every Nth frame in second pass
    DETECTION_MARGIN = 2.0  # seconds to add before/after detected segments
    MIN_DETECTION_CONFIDENCE = 0.35
    MAX_BATCH_SIZE = 32

config = Config()

class TwoPassProcessor:
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

    def first_pass(self, video_path: str) -> List[Dict[str, Any]]:
        """First pass: Quick YOLO detection to identify potential segments"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        
        potential_segments = []
        current_segment = None
        
        try:
            while cap.isOpened() and frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every Nth frame in first pass
                if frame_idx % config.SAMPLE_RATE_FIRST_PASS == 0:
                    # Run YOLO detection
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.yolo(frame_rgb, verbose=False)[0]
                    
                    # If any detections found
                    if len(results.boxes) > 0:
                        timestamp = frame_idx / fps
                        
                        if current_segment is None:
                            current_segment = {
                                'start_time': max(0, timestamp - config.DETECTION_MARGIN),
                                'end_time': timestamp
                            }
                        else:
                            current_segment['end_time'] = timestamp
                    
                    elif current_segment is not None:
                        # End current segment
                        current_segment['end_time'] += config.DETECTION_MARGIN
                        potential_segments.append(current_segment)
                        current_segment = None
                
                frame_idx += 1
            
            # Handle last segment
            if current_segment is not None:
                current_segment['end_time'] = min(
                    current_segment['end_time'] + config.DETECTION_MARGIN,
                    total_frames / fps
                )
                potential_segments.append(current_segment)
            
            return potential_segments
            
        except Exception as e:
            logger.error(f"Error in first pass: {str(e)}")
            raise
        finally:
            cap.release()

    def process_segment(self, frames: List[np.ndarray], prompt: str) -> List[Dict[str, Any]]:
        """Process a segment of frames with CLIP"""
        try:
            detections = []
            
            for frame in frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Run YOLO detection
                results = self.yolo(frame_rgb, verbose=False)[0]
                
                # Process each detection
                frame_detections = []
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
                            frame_detections.append(box)
                
                if frame_detections:
                    detections.append(frame_detections)
                else:
                    detections.append([])
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing segment: {str(e)}")
            return [[] for _ in frames]
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
            
            for frame in frames:
                out.write(frame)
            
            out.release()
            return True
            
        except Exception as e:
            logger.error(f"Error saving preview: {str(e)}")
            return False

    def process_video(self, video_path: str, prompt: str) -> List[Dict[str, Any]]:
        """Process video using two-pass approach"""
        try:
            # First pass: identify potential segments
            logger.info("Starting first pass...")
            potential_segments = self.first_pass(video_path)
            logger.info(f"First pass complete. Found {len(potential_segments)} potential segments")
            
            if not potential_segments:
                return []
            
            # Second pass: detailed analysis of potential segments
            logger.info("Starting second pass...")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            final_segments = []
            
            try:
                for segment in potential_segments:
                    start_frame = int(segment['start_time'] * fps)
                    end_frame = int(segment['end_time'] * fps)
                    
                    # Skip to segment start
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    
                    # Collect frames for this segment
                    segment_frames = []
                    frame_idx = start_frame
                    
                    while frame_idx < end_frame:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        if (frame_idx - start_frame) % config.SAMPLE_RATE_SECOND_PASS == 0:
                            segment_frames.append(frame)
                        
                        frame_idx += 1
                    
                    if segment_frames:
                        # Process segment frames
                        detections = self.process_segment(segment_frames, prompt)
                        
                        # Create final segment if detections found
                        current_segment = None
                        
                        for i, frame_detections in enumerate(detections):
                            if frame_detections:
                                timestamp = (start_frame + i * config.SAMPLE_RATE_SECOND_PASS) / fps
                                
                                if current_segment is None:
                                    current_segment = {
                                        'start_time': timestamp,
                                        'end_time': timestamp,
                                        'preview_frames': [segment_frames[i]],
                                        'max_confidence': max(d['confidence'] for d in frame_detections),
                                        'max_similarity': max(d['similarity'] for d in frame_detections)
                                    }
                                else:
                                    current_segment['end_time'] = timestamp
                                    current_segment['preview_frames'].append(segment_frames[i])
                                    current_segment['max_confidence'] = max(
                                        current_segment['max_confidence'],
                                        max(d['confidence'] for d in frame_detections)
                                    )
                                    current_segment['max_similarity'] = max(
                                        current_segment['max_similarity'],
                                        max(d['similarity'] for d in frame_detections)
                                    )
                            
                            elif current_segment is not None:
                                # End current segment if it meets minimum duration
                                if current_segment['end_time'] - current_segment['start_time'] >= config.MIN_SEGMENT_DURATION:
                                    preview_path = PREVIEW_DIR / f"segment_{len(final_segments)}_{int(current_segment['start_time'])}_{int(current_segment['end_time'])}.mp4"
                                    if self.save_preview(current_segment['preview_frames'], preview_path):
                                        segment_data = {
                                            'start_time': float(current_segment['start_time']),
                                            'end_time': float(current_segment['end_time']),
                                            'confidence': float(current_segment['max_confidence']),
                                            'similarity': float(current_segment['max_similarity']),
                                            'video_preview': str(preview_path.relative_to(UPLOAD_DIR.parent))
                                        }
                                        final_segments.append(segment_data)
                                
                                current_segment = None
                        
                        # Handle last segment
                        if current_segment is not None and current_segment['end_time'] - current_segment['start_time'] >= config.MIN_SEGMENT_DURATION:
                            preview_path = PREVIEW_DIR / f"segment_{len(final_segments)}_{int(current_segment['start_time'])}_{int(current_segment['end_time'])}.mp4"
                            if self.save_preview(current_segment['preview_frames'], preview_path):
                                segment_data = {
                                    'start_time': float(current_segment['start_time']),
                                    'end_time': float(current_segment['end_time']),
                                    'confidence': float(current_segment['max_confidence']),
                                    'similarity': float(current_segment['max_similarity']),
                                    'video_preview': str(preview_path.relative_to(UPLOAD_DIR.parent))
                                }
                                final_segments.append(segment_data)
            
            finally:
                cap.release()
            
            logger.info(f"Second pass complete. Found {len(final_segments)} final segments")
            return final_segments
            
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
processor = TwoPassProcessor()

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