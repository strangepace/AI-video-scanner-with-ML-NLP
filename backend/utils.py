import cv2
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from datetime import timedelta

def format_timestamp(seconds: float) -> str:
    """
    Format seconds into HH:MM:SS format
    """
    return str(timedelta(seconds=int(seconds)))

def get_video_info(video_path: str) -> Dict:
    """
    Get basic information about a video file
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    }
    
    cap.release()
    return info

def extract_frame_preview(frame: np.ndarray, max_size: Tuple[int, int] = (320, 240)) -> np.ndarray:
    """
    Extract a preview frame and resize it to fit within max_size while maintaining aspect ratio
    """
    height, width = frame.shape[:2]
    target_width, target_height = max_size
    
    # Calculate aspect ratio
    aspect = width / height
    
    if width > height:
        new_width = target_width
        new_height = int(target_width / aspect)
    else:
        new_height = target_height
        new_width = int(target_height * aspect)
    
    # Resize frame
    resized = cv2.resize(frame, (new_width, new_height))
    
    # Convert to JPEG format
    _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    return buffer.tobytes()

def merge_overlapping_segments(segments: List[Dict], overlap_threshold: float = 1.0) -> List[Dict]:
    """
    Merge overlapping video segments
    """
    if not segments:
        return []
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x['timestamp'])
    merged = []
    current = sorted_segments[0]
    
    for next_seg in sorted_segments[1:]:
        if next_seg['timestamp'] - current['timestamp'] <= overlap_threshold:
            # Merge segments
            current['end_timestamp'] = max(current.get('end_timestamp', current['timestamp']), 
                                        next_seg.get('end_timestamp', next_seg['timestamp']))
        else:
            merged.append(current)
            current = next_seg
    
    merged.append(current)
    return merged

def create_segment_preview(video_path: str, start_time: float, end_time: float) -> bytes:
    """
    Create a preview image for a video segment
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    mid_frame = int((start_time + end_time) / 2 * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Could not extract preview frame")

    return extract_frame_preview(frame) 