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
from typing import List, Dict, Any, Tuple, Queue
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
import multiprocessing as mp
from queue import Empty
from threading import Thread
import time

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
    BATCH_SIZE = 16  # frames per batch
    MIN_DETECTION_CONFIDENCE = 0.35
    MAX_QUEUE_SIZE = 50
    FRAME_BUFFER_SIZE = 300  # maximum frames to hold in memory

config = Config()

class ParallelProcessor:
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

    def detection_worker(self, frame_queue: mp.Queue, detection_queue: mp.Queue, stop_event: mp.Event):
        """Worker process for object detection"""
        try:
            while not stop_event.is_set():
                try:
                    frame_data = frame_queue.get(timeout=1)
                    if frame_data is None:
                        break
                    
                    frame, frame_idx = frame_data
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Run YOLO detection
                    results = self.yolo(frame_rgb, verbose=False)[0]
                    detections = []
                    
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        # Ensure valid coordinates
                        x1, x2 = max(0, x1), min(frame.shape[1], x2)
                        y1, y2 = max(0, y1), min(frame.shape[0], y2)
                        
                        if x2 > x1 and y2 > y1:
                            crop = frame_rgb[y1:y2, x1:x2]
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'class_name': self.yolo.model.names[cls],
                                'crop': crop
                            })
                    
                    if detections:
                        detection_queue.put((frame_idx, frame, detections))
                    
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in detection worker: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Detection worker failed: {str(e)}")
        finally:
            logger.info("Detection worker stopped")

    def matching_worker(self, detection_queue: mp.Queue, result_queue: mp.Queue, 
                       prompt: str, stop_event: mp.Event):
        """Worker process for CLIP matching"""
        try:
            while not stop_event.is_set():
                try:
                    data = detection_queue.get(timeout=1)
                    if data is None:
                        break
                    
                    frame_idx, frame, detections = data
                    matched_detections = []
                    
                    # Convert crops to PIL images
                    crops = [Image.fromarray(d['crop']) for d in detections]
                    
                    if crops:
                        # Batch process with CLIP
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
                        for det, sim in zip(detections, similarities):
                            if sim > config.SIMILARITY_THRESHOLD:
                                det_copy = det.copy()
                                del det_copy['crop']  # Remove image data
                                det_copy['similarity'] = float(sim)
                                matched_detections.append(det_copy)
                    
                    if matched_detections:
                        result_queue.put((frame_idx, frame, matched_detections))
                    
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in matching worker: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Matching worker failed: {str(e)}")
        finally:
            logger.info("Matching worker stopped")
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
        """Process video using parallel workers"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize queues and events
            frame_queue = mp.Queue(maxsize=config.MAX_QUEUE_SIZE)
            detection_queue = mp.Queue(maxsize=config.MAX_QUEUE_SIZE)
            result_queue = mp.Queue()
            stop_event = mp.Event()
            
            # Start workers
            detection_process = Thread(
                target=self.detection_worker,
                args=(frame_queue, detection_queue, stop_event)
            )
            matching_process = Thread(
                target=self.matching_worker,
                args=(detection_queue, result_queue, prompt, stop_event)
            )
            
            detection_process.start()
            matching_process.start()
            
            # Process frames
            frame_idx = 0
            segments = []
            current_segment = None
            segment_frames = []
            
            try:
                while cap.isOpened() and frame_idx < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Add frame to queue
                    frame_queue.put((frame.copy(), frame_idx))
                    
                    # Check for results
                    try:
                        while True:
                            result_idx, result_frame, detections = result_queue.get_nowait()
                            timestamp = result_idx / fps
                            
                            if current_segment is None:
                                current_segment = {
                                    'start_time': timestamp,
                                    'end_time': timestamp,
                                    'preview_frames': [result_frame],
                                    'max_confidence': max(d['confidence'] for d in detections),
                                    'max_similarity': max(d['similarity'] for d in detections)
                                }
                            else:
                                # Check if this result belongs to current segment
                                if timestamp - current_segment['end_time'] <= config.MIN_SEGMENT_DURATION:
                                    current_segment['end_time'] = timestamp
                                    current_segment['preview_frames'].append(result_frame)
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
                                        'preview_frames': [result_frame],
                                        'max_confidence': max(d['confidence'] for d in detections),
                                        'max_similarity': max(d['similarity'] for d in detections)
                                    }
                            
                    except Empty:
                        pass
                    
                    frame_idx += 1
                    
                    # Clear GPU memory periodically
                    if frame_idx % (config.BATCH_SIZE * 10) == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                
                # Signal workers to stop
                stop_event.set()
                frame_queue.put(None)
                detection_queue.put(None)
                
                # Wait for workers to finish
                detection_process.join()
                matching_process.join()
                
                # Process remaining results
                while True:
                    try:
                        result_idx, result_frame, detections = result_queue.get_nowait()
                        timestamp = result_idx / fps
                        
                        if current_segment is None:
                            current_segment = {
                                'start_time': timestamp,
                                'end_time': timestamp,
                                'preview_frames': [result_frame],
                                'max_confidence': max(d['confidence'] for d in detections),
                                'max_similarity': max(d['similarity'] for d in detections)
                            }
                        else:
                            if timestamp - current_segment['end_time'] <= config.MIN_SEGMENT_DURATION:
                                current_segment['end_time'] = timestamp
                                current_segment['preview_frames'].append(result_frame)
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
                                
                                current_segment = {
                                    'start_time': timestamp,
                                    'end_time': timestamp,
                                    'preview_frames': [result_frame],
                                    'max_confidence': max(d['confidence'] for d in detections),
                                    'max_similarity': max(d['similarity'] for d in detections)
                                }
                                
                    except Empty:
                        break
                
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
processor = ParallelProcessor()

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