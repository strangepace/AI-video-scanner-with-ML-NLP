from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import torch

class VideoDetector:
    def __init__(self):
        # Initialize YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # Using the nano model for speed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def process_frame(self, frame: np.ndarray, conf_threshold: float = 0.3) -> List[Dict]:
        """
        Process a single frame and return detected objects
        """
        results = self.model(frame, verbose=False, conf=conf_threshold)[0]
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': confidence,
                'class': class_name
            })
        
        return detections

    def process_frames_batch(self, frames: List[np.ndarray], conf_threshold: float = 0.3) -> List[List[Dict]]:
        """
        Process a batch of frames for better GPU utilization
        """
        results = self.model(frames, verbose=False, conf=conf_threshold)
        all_detections = []
        
        for result in results:
            frame_detections = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                frame_detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': confidence,
                    'class': class_name
                })
            all_detections.append(frame_detections)
        
        return all_detections

    def process_video(self, video_path: str, sample_rate: int = 1, 
                     conf_threshold: float = 0.3, batch_size: int = 4) -> List[Dict]:
        """
        Process video file and return detections with timestamps
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        detections_with_time = []

        # For batched processing
        batch_frames = []
        batch_timestamps = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every nth frame based on sample_rate
            if frame_count % sample_rate == 0:
                timestamp = frame_count / fps
                batch_frames.append(frame)
                batch_timestamps.append(timestamp)
                
                # When batch is full, process it
                if len(batch_frames) >= batch_size:
                    batch_detections = self.process_frames_batch(batch_frames, conf_threshold)
                    
                    # Add results to detections list
                    for i, (ts, frame_detections) in enumerate(zip(batch_timestamps, batch_detections)):
                        if frame_detections:  # Only add frames with detections
                            detections_with_time.append({
                                'timestamp': ts,
                                'detections': frame_detections
                            })
                    
                    # Clear batch
                    batch_frames = []
                    batch_timestamps = []

            frame_count += 1

        # Process any remaining frames in the last batch
        if batch_frames:
            batch_detections = self.process_frames_batch(batch_frames, conf_threshold)
            for i, (ts, frame_detections) in enumerate(zip(batch_timestamps, batch_detections)):
                if frame_detections:
                    detections_with_time.append({
                        'timestamp': ts,
                        'detections': frame_detections
                    })

        cap.release()
        return detections_with_time

    def get_frame_at_timestamp(self, video_path: str, timestamp: float) -> np.ndarray:
        """
        Extract a specific frame from the video at the given timestamp
        """
        cap = cv2.VideoCapture(str(video_path))
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