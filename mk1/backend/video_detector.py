import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from deep_sort_realtime.deepsort_tracker import DeepSort
from skimage.metrics import structural_similarity as ssim
import os
from tqdm import tqdm
import time

class AdvancedVideoDetector:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize YOLO model
        self.yolo_model = YOLO(self.config['yolo']['model_path'])
        self.yolo_model.to(self.device)
        
        # Initialize CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)
        
        # Initialize DeepSORT tracker with supported parameters
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True
        )
        
        # Create necessary directories
        os.makedirs(self.config['general']['model_dir'], exist_ok=True)
        os.makedirs(self.config['general']['upload_dir'], exist_ok=True)

    def process_frame(self, frame, prompt):
        """Process a single frame with YOLO and CLIP"""
        # Run YOLO detection
        results = self.yolo_model(frame, verbose=False)[0]
        
        matches = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            if conf < self.config['yolo']['confidence_threshold']:
                continue
                
            # Crop the detected object
            obj_img = frame[y1:y2, x1:x2]
            if obj_img.size == 0:
                continue
                
            # Convert to PIL Image for CLIP
            obj_pil = Image.fromarray(cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB))
            
            # Process with CLIP
            inputs = self.clip_processor(
                text=[prompt],
                images=obj_pil,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                similarity = torch.sigmoid(logits_per_image).item()
            
            if similarity > self.config['clip']['similarity_threshold']:
                matches.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'similarity': similarity,
                    'class_id': cls,
                    'class_name': results.names[cls]
                })
        
        return matches

    def track_objects(self, frame):
        """Track objects in a frame using DeepSORT"""
        results = self.yolo_model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            if conf < self.config['yolo']['confidence_threshold']:
                continue
                
            detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))
        
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks

    def detect_motion(self, frame, prev_frame):
        """Detect motion between two frames"""
        if prev_frame is None:
            return True
            
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate structural similarity
        score, _ = ssim(gray1, gray2, full=True)
        return score < self.config['motion']['threshold']

    def process_video(self, video_path, prompt, mode='full_scan'):
        """Process video based on selected mode"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = self.config['frame_skip']['skip_frames']
        
        results = []
        prev_frame = None
        frame_count = 0
        
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Skip frames based on mode
                if mode == 'frame_skip' and frame_count % frame_skip != 0:
                    pbar.update(1)
                    continue
                    
                if mode == 'motion_filter':
                    if not self.detect_motion(frame, prev_frame):
                        prev_frame = frame.copy()
                        pbar.update(1)
                        continue
                
                # Process frame
                if mode in ['full_scan', 'frame_skip', 'motion_filter']:
                    matches = self.process_frame(frame, prompt)
                    for match in matches:
                        timestamp = frame_count / fps
                        results.append({
                            'timestamp': timestamp,
                            'frame': frame_count,
                            **match
                        })
                
                elif mode == 'track_then_match':
                    try:
                        tracks = self.track_objects(frame)
                        for track in tracks:
                            if not track.is_confirmed():
                                continue
                                
                            ltrb = track.to_ltrb()
                            x1, y1, x2, y2 = map(int, ltrb)
                            obj_img = frame[y1:y2, x1:x2]
                            
                            if obj_img.size == 0:
                                continue
                                
                            obj_pil = Image.fromarray(cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB))
                            inputs = self.clip_processor(
                                text=[prompt],
                                images=obj_pil,
                                return_tensors="pt",
                                padding=True
                            ).to(self.device)
                            
                            with torch.no_grad():
                                outputs = self.clip_model(**inputs)
                                similarity = torch.sigmoid(outputs.logits_per_image).item()
                            
                            if similarity > self.config['clip']['similarity_threshold']:
                                timestamp = frame_count / fps
                                results.append({
                                    'timestamp': timestamp,
                                    'frame': frame_count,
                                    'bbox': (x1, y1, x2, y2),
                                    'track_id': track.track_id,
                                    'similarity': similarity
                                })
                    except Exception as e:
                        print(f"Warning: Tracking error in frame {frame_count}: {str(e)}")
                        continue
                
                prev_frame = frame.copy()
                pbar.update(1)
        
        cap.release()
        return results 