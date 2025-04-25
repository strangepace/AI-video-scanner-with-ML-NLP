from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
from typing import List, Dict, Any

class VideoProcessor:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # Load YOLOv8 model
        
    def process_video(self, video_path: str, search_query: str) -> List[Dict[str, Any]]:
        """
        Process video and search for objects matching the query
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        results = []
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every 5th frame for efficiency
            if frame_count % 5 == 0:
                # Run YOLOv8 inference on the frame
                detections = self.model(frame)[0]
                
                # Process detections
                for detection in detections.boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = detection
                    class_name = self.model.names[int(class_id)]
                    
                    # Simple text matching for now - can be enhanced with NLP
                    if search_query.lower() in class_name.lower():
                        timestamp = frame_count / fps
                        results.append({
                            "timestamp": timestamp,
                            "confidence": conf,
                            "class": class_name,
                            "bbox": [x1, y1, x2, y2]
                        })
            
            frame_count += 1
            
        cap.release()
        return results
        
    def get_frame(self, video_path: str, timestamp: float) -> str:
        """
        Extract a frame from the video at the given timestamp
        """
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not extract frame at timestamp {timestamp}")
            
        # Save frame as temporary image
        frame_path = Path("uploads") / f"frame_{timestamp}.jpg"
        cv2.imwrite(str(frame_path), frame)
        return str(frame_path) 