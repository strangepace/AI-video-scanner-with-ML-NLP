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
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

class Config:
    SIMILARITY_THRESHOLD = 0.25
    MIN_SEGMENT_DURATION = 1.0  # seconds
    BATCH_SIZE = 32  # frames per batch
    CHECKPOINT_INTERVAL = 100  # save checkpoint every N frames
    MIN_DETECTION_CONFIDENCE = 0.35
    MAX_MEMORY_FRAMES = 500  # maximum frames to hold in memory

config = Config()

class BatchProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.yolo = YOLO('yolov8m.pt')
        self.yolo.conf = config.MIN_DETECTION_CONFIDENCE
        logger.info("Loaded YOLOv8 model")
        
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        logger.info("Loaded CLIP model")

    def process_batch(self, frames: List[np.ndarray], prompt: str) -> List[Dict[str, Any]]:
        """Process a batch of frames and return detections"""
        batch_results = []
        
        try:
            for frame in frames:
                # Convert frame to RGB for YOLO
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Run YOLO detection
                results = self.yolo(frame_rgb, verbose=False)[0]
                frame_detections = []
                crops = []
                
                # Collect all crops for batch processing
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
                        frame_detections.append({
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
                    
                    # Add similarity scores to detections
                    matched_detections = []
                    for det, sim in zip(frame_detections, similarities):
                        if sim > config.SIMILARITY_THRESHOLD:
                            det['similarity'] = float(sim)
                            matched_detections.append(det)
                    
                    if matched_detections:
                        batch_results.append(matched_detections)
                    else:
                        batch_results.append([])
                else:
                    batch_results.append([])
                
            return batch_results
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return [[] for _ in frames]  # Return empty results for all frames
        finally:
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()

    def save_checkpoint(self, checkpoint_data: Dict[str, Any], file_id: str, batch_num: int):
        """Save processing checkpoint"""
        try:
            checkpoint_path = CHECKPOINT_DIR / f"{file_id}_batch_{batch_num}.json"
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")

    def load_checkpoint(self, file_id: str) -> Tuple[int, List[Dict[str, Any]]]:
        """Load latest checkpoint if exists"""
        try:
            checkpoints = list(CHECKPOINT_DIR.glob(f"{file_id}_batch_*.json"))
            if not checkpoints:
                return 0, []
            
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            with open(latest_checkpoint, 'r') as f:
                checkpoint_data = json.load(f)
            
            batch_num = checkpoint_data.get('batch_num', 0)
            segments = checkpoint_data.get('segments', [])
            
            logger.info(f"Loaded checkpoint: {latest_checkpoint}")
            return batch_num, segments
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return 0, []

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

    def process_video(self, video_path: str, prompt: str, file_id: str) -> List[Dict[str, Any]]:
        """Process video in batches with checkpointing"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Load checkpoint if exists
        start_batch, segments = self.load_checkpoint(file_id)
        frame_idx = start_batch * config.BATCH_SIZE
        
        current_batch = []
        current_segment = None
        preview_frames = []
        
        try:
            # Skip to the correct frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            while cap.isOpened() and frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_batch.append(frame)
                
                # Process batch when full
                if len(current_batch) >= config.BATCH_SIZE:
                    batch_results = self.process_batch(current_batch, prompt)
                    
                    # Process results frame by frame
                    for i, detections in enumerate(batch_results):
                        timestamp = (frame_idx - len(current_batch) + i + 1) / fps
                        
                        if detections:
                            if current_segment is None:
                                # Start new segment
                                current_segment = {
                                    'start_time': timestamp,
                                    'end_time': timestamp,
                                    'preview_frames': [current_batch[i]],
                                    'max_confidence': max(d['confidence'] for d in detections),
                                    'max_similarity': max(d['similarity'] for d in detections)
                                }
                            else:
                                # Update existing segment
                                current_segment['end_time'] = timestamp
                                current_segment['preview_frames'].append(current_batch[i])
                                current_segment['max_confidence'] = max(
                                    current_segment['max_confidence'],
                                    max(d['confidence'] for d in detections)
                                )
                                current_segment['max_similarity'] = max(
                                    current_segment['max_similarity'],
                                    max(d['similarity'] for d in detections)
                                )
                        
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
                    
                    # Save checkpoint
                    if frame_idx % config.CHECKPOINT_INTERVAL == 0:
                        checkpoint_data = {
                            'batch_num': frame_idx // config.BATCH_SIZE,
                            'segments': segments
                        }
                        self.save_checkpoint(checkpoint_data, file_id, frame_idx // config.BATCH_SIZE)
                    
                    # Clear batch
                    current_batch = []
                    
                    # Clear GPU memory periodically
                    if frame_idx % (config.BATCH_SIZE * 10) == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                
                frame_idx += 1
            
            # Process remaining frames
            if current_batch:
                batch_results = self.process_batch(current_batch, prompt)
                
                for i, detections in enumerate(batch_results):
                    timestamp = (frame_idx - len(current_batch) + i + 1) / fps
                    
                    if detections:
                        if current_segment is None:
                            current_segment = {
                                'start_time': timestamp,
                                'end_time': timestamp,
                                'preview_frames': [current_batch[i]],
                                'max_confidence': max(d['confidence'] for d in detections),
                                'max_similarity': max(d['similarity'] for d in detections)
                            }
                        else:
                            current_segment['end_time'] = timestamp
                            current_segment['preview_frames'].append(current_batch[i])
                            current_segment['max_confidence'] = max(
                                current_segment['max_confidence'],
                                max(d['confidence'] for d in detections)
                            )
                            current_segment['max_similarity'] = max(
                                current_segment['max_similarity'],
                                max(d['similarity'] for d in detections)
                            )
            
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
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise
        finally:
            cap.release()
            torch.cuda.empty_cache()
            gc.collect()

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
batch_processor = BatchProcessor()

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
        segments = batch_processor.process_video(str(file_path), prompt, file_id)
        
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