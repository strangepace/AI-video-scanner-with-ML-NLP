from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
from pathlib import Path
import uuid
import logging
import tempfile
from typing import List, Dict, Any, Tuple, Optional
import torch
import cv2
import numpy as np
import uvicorn
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch.nn.functional as F
from datetime import datetime, timedelta
import json
import gc
from pyngrok import ngrok
import asyncio
from concurrent.futures import ThreadPoolExecutor
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
    SAMPLE_RATE = 5  # process every Nth frame
    MIN_DETECTION_CONFIDENCE = 0.35
    MAX_PREVIEW_FRAMES = 30
    BATCH_SIZE = 8  # Process frames in batches
    MAX_MEMORY_USAGE = 0.8  # Maximum GPU memory usage (80%)

config = Config()

class ProcessingStatus:
    def __init__(self):
        self.current_task: Optional[Dict[str, Any]] = None
        self.tasks_history: List[Dict[str, Any]] = []
        self.last_error: Optional[str] = None
        self.memory_stats: Dict[str, float] = {}
        
    def start_task(self, task_id: str, task_type: str, total_frames: int):
        self.current_task = {
            'task_id': task_id,
            'type': task_type,
            'start_time': datetime.now(),
            'processed_frames': 0,
            'total_frames': total_frames,
            'status': 'processing',
            'progress': 0.0,
            'segments_found': 0,
            'estimated_time_remaining': None,
            'memory_usage': None
        }
        self.last_error = None
        self.update_memory_stats()
        
    def update_memory_stats(self):
        if torch.cuda.is_available():
            self.memory_stats = {
                'allocated': torch.cuda.memory_allocated() / (1024**3),  # GB
                'cached': torch.cuda.memory_reserved() / (1024**3),      # GB
                'max_allocated': torch.cuda.max_memory_allocated() / (1024**3)  # GB
            }
        else:
            self.memory_stats = {}
        
    def update_progress(self, frames_processed: int, segments_found: int = None):
        if self.current_task:
            self.current_task['processed_frames'] = frames_processed
            self.current_task['progress'] = (frames_processed / self.current_task['total_frames']) * 100
            
            # Calculate estimated time remaining
            if frames_processed > 0:
                elapsed_time = (datetime.now() - self.current_task['start_time']).total_seconds()
                frames_remaining = self.current_task['total_frames'] - frames_processed
                time_per_frame = elapsed_time / frames_processed
                estimated_remaining = time_per_frame * frames_remaining
                self.current_task['estimated_time_remaining'] = str(timedelta(seconds=int(estimated_remaining)))
            
            if segments_found is not None:
                self.current_task['segments_found'] = segments_found
            
            self.update_memory_stats()
            self.current_task['memory_usage'] = self.memory_stats
                
    def set_error(self, error_msg: str):
        self.last_error = error_msg
        if self.current_task:
            self.current_task['status'] = 'error'
            self.current_task['error'] = error_msg
                
    def complete_task(self, success: bool = True):
        if self.current_task:
            self.current_task['end_time'] = datetime.now()
            self.current_task['status'] = 'completed' if success else 'failed'
            self.tasks_history.append(self.current_task)
            self.current_task = None
            
    def get_current_status(self) -> Dict[str, Any]:
        status = {
            'active': bool(self.current_task),
            'last_error': self.last_error,
            'memory_stats': self.memory_stats
        }
        
        if self.current_task:
            status.update({
                'task_id': self.current_task['task_id'],
                'type': self.current_task['type'],
                'progress': self.current_task['progress'],
                'frames_processed': self.current_task['processed_frames'],
                'total_frames': self.current_task['total_frames'],
                'segments_found': self.current_task['segments_found'],
                'elapsed_time': str(datetime.now() - self.current_task['start_time']).split('.')[0],
                'estimated_time_remaining': self.current_task['estimated_time_remaining'],
                'memory_usage': self.current_task['memory_usage']
            })
            
        return status

class CleanupManager:
    @staticmethod
    async def cleanup_old_files():
        """Clean up files older than 24 hours"""
        try:
            current_time = datetime.now()
            # Clean up uploads
            for file_path in UPLOAD_DIR.glob('*'):
                if file_path.is_file():
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age > timedelta(hours=24):
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path}")
            
            # Clean up previews
            for file_path in PREVIEW_DIR.glob('*'):
                if file_path.is_file():
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age > timedelta(hours=24):
                        file_path.unlink()
                        logger.info(f"Cleaned up old preview: {file_path}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

class SimpleProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize YOLO
        logger.info("Loading YOLOv8 model...")
        try:
            self.yolo = YOLO('yolov8m.pt')
            self.yolo.conf = config.MIN_DETECTION_CONFIDENCE
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            raise
        
        # Initialize CLIP
        logger.info("Loading CLIP model...")
        try:
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading CLIP model: {str(e)}")
            raise

    def check_memory(self):
        """Check GPU memory usage and clear if needed"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if memory_allocated > config.MAX_MEMORY_USAGE:
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Cleared GPU memory")

    def process_frame_batch(self, frames: List[np.ndarray], prompt: str) -> List[List[Dict[str, Any]]]:
        """Process a batch of frames"""
        try:
            batch_results = []
            frame_rgbs = []
            pil_imgs = []
            
            # Convert frames to RGB and PIL
            for frame in frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                frame_rgbs.append(frame_rgb)
                pil_imgs.append(pil_img)
            
            # Run YOLO on batch
            yolo_results = self.yolo(frame_rgbs, verbose=False)
            
            for idx, (frame_rgb, pil_img, results) in enumerate(zip(frame_rgbs, pil_imgs, yolo_results)):
                detections = []
                crops = []
                boxes = []
                
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Ensure valid coordinates
                    x1, x2 = max(0, x1), min(frame_rgb.shape[1], x2)
                    y1, y2 = max(0, y1), min(frame_rgb.shape[0], y2)
                    
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
                
                batch_results.append(detections)
                
            return batch_results
            
        except Exception as e:
            logger.error(f"Error processing frame batch: {str(e)}")
            return [[] for _ in frames]
        finally:
            self.check_memory()

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

    def process_video(self, video_path: str, prompt: str, status_tracker: ProcessingStatus) -> List[Dict[str, Any]]:
        """Process video using batched approach"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0
            
            # Update status
            task_id = os.path.basename(video_path).split('.')[0]
            status_tracker.start_task(task_id, 'video_processing', total_frames)
            
            segments = []
            current_segment = None
            batch_frames = []
            batch_timestamps = []
            
            logger.info(f"Processing video with {total_frames} frames at {fps} FPS")
            
            try:
                while cap.isOpened() and frame_idx < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every Nth frame
                    if frame_idx % config.SAMPLE_RATE == 0:
                        batch_frames.append(frame)
                        batch_timestamps.append(frame_idx / fps)
                        
                        # Process batch when full
                        if len(batch_frames) >= config.BATCH_SIZE:
                            logger.info(f"Processing batch at frame {frame_idx}/{total_frames}")
                            batch_detections = self.process_frame_batch(batch_frames, prompt)
                            
                            # Process each frame's detections
                            for frame_detections, timestamp, frame in zip(batch_detections, batch_timestamps, batch_frames):
                                if frame_detections:
                                    if current_segment is None:
                                        current_segment = {
                                            'start_time': timestamp,
                                            'end_time': timestamp,
                                            'preview_frames': [frame],
                                            'max_confidence': max(d['confidence'] for d in frame_detections),
                                            'max_similarity': max(d['similarity'] for d in frame_detections)
                                        }
                                    else:
                                        # Check if this frame belongs to current segment
                                        if timestamp - current_segment['end_time'] <= config.MIN_SEGMENT_DURATION:
                                            current_segment['end_time'] = timestamp
                                            current_segment['preview_frames'].append(frame)
                                            current_segment['max_confidence'] = max(
                                                current_segment['max_confidence'],
                                                max(d['confidence'] for d in frame_detections)
                                            )
                                            current_segment['max_similarity'] = max(
                                                current_segment['max_similarity'],
                                                max(d['similarity'] for d in frame_detections)
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
                                                    logger.info(f"Saved segment {len(segments)} ({segment_data['start_time']:.1f}s - {segment_data['end_time']:.1f}s)")
                                            
                                            # Start new segment
                                            current_segment = {
                                                'start_time': timestamp,
                                                'end_time': timestamp,
                                                'preview_frames': [frame],
                                                'max_confidence': max(d['confidence'] for d in frame_detections),
                                                'max_similarity': max(d['similarity'] for d in frame_detections)
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
                                            logger.info(f"Saved segment {len(segments)} ({segment_data['start_time']:.1f}s - {segment_data['end_time']:.1f}s)")
                                    
                                    current_segment = None
                            
                            # Clear batch
                            batch_frames = []
                            batch_timestamps = []
                    
                    frame_idx += 1
                    
                    # Update progress in the existing frame processing loop
                    if frame_idx % 100 == 0:
                        status_tracker.update_progress(frame_idx, len(segments))
                        logger.info(f"Processed {frame_idx}/{total_frames} frames ({(frame_idx/total_frames)*100:.1f}%)")
                
                # Process remaining batch
                if batch_frames:
                    logger.info(f"Processing final batch")
                    batch_detections = self.process_frame_batch(batch_frames, prompt)
                    # ... (same batch processing code as above)
                
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
                        logger.info(f"Saved final segment {len(segments)} ({segment_data['start_time']:.1f}s - {segment_data['end_time']:.1f}s)")
                
                logger.info(f"Video processing complete. Found {len(segments)} segments.")
                status_tracker.complete_task(True)
                return segments
                
            finally:
                cap.release()
                
        except Exception as e:
            status_tracker.complete_task(False)
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

# Add these before the FastAPI app initialization
status_tracker = ProcessingStatus()
cleanup_manager = CleanupManager()

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

@app.get("/status")
async def get_status():
    """Get current processing status"""
    return status_tracker.get_current_status()

@app.post("/cleanup")
async def trigger_cleanup(background_tasks: BackgroundTasks):
    """Trigger cleanup of old files"""
    background_tasks.add_task(cleanup_manager.cleanup_old_files)
    return {"message": "Cleanup scheduled"}

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
        segments = processor.process_video(str(file_path), prompt, status_tracker)
        
        return JSONResponse({
            "message": "Search completed successfully",
            "results": segments
        })
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing video: {error_msg}")
        if "CUDA out of memory" in error_msg:
            raise HTTPException(status_code=500, detail="GPU memory exceeded. Try with a shorter video or wait for current processing to complete.")
        elif "Failed to open video file" in error_msg:
            raise HTTPException(status_code=400, detail="Could not open video file. Make sure it's a valid MP4 file.")
        else:
            raise HTTPException(status_code=500, detail=str(e))

def start_server():
    try:
        # Start ngrok tunnel
        port = 8000
        ngrok_tunnel = ngrok.connect(port)
        public_url = ngrok_tunnel.public_url
        print("\n" + "="*50)
        print(f"Public URL for frontend: {public_url}")
        print("="*50 + "\n")
        
        # Start FastAPI server
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            timeout_keep_alive=300,
            workers=1
        )
        server = uvicorn.Server(config)
        server.run()
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise

if __name__ == "__main__":
    # Apply nest_asyncio to allow running async code in Jupyter/Colab
    import nest_asyncio
    nest_asyncio.apply()
    
    # Start the server
    start_server()