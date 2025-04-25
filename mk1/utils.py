import cv2
import numpy as np
from typing import Dict, Tuple
import time

def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def get_video_info(video_path: str) -> Dict:
    """Get video information"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }

def extract_frame_preview(frame: np.ndarray, size: Tuple[int, int] = (320, 240)) -> bytes:
    """Extract and resize frame preview"""
    # Resize frame
    resized = cv2.resize(frame, size)
    
    # Convert to JPEG
    _, buffer = cv2.imencode('.jpg', resized)
    
    return buffer.tobytes()

def get_frame_at_timestamp(video_path: str, timestamp: float) -> np.ndarray:
    """Get frame at specific timestamp"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not extract frame at timestamp {timestamp}")
    
    return frame 