import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
import yaml
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from clip_matcher import CLIPMatcher

class AdvancedVideoDetector:
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize YOLOv8 model
        self.yolo = YOLO(self.config['yolo']['model'])
        self.device = self.config['yolo']['device']
        self.yolo.to(self.device)
        
        # Initialize CLIP matcher
        self.clip = CLIPMatcher()
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=self.config['modes']['track_then_match']['max_age'],
            n_init=self.config['modes']['track_then_match']['min_hits']
        )
        
        # Initialize background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            detectShadows=False
        )

    def detect_motion(self, frame: np.ndarray) -> bool:
        """Detect motion in the frame using background subtraction"""
        fg_mask = self.bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, self.config['modes']['motion_filter']['motion_threshold'], 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any contour is large enough
        for contour in contours:
            if cv2.contourArea(contour) > self.config['modes']['motion_filter']['min_contour_area']:
                return True
        return False

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Process a single frame with YOLOv8"""
        results = self.yolo(frame, verbose=False, conf=self.config['yolo']['confidence_threshold'])[0]
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

    def process_video_full_scan(self, video_path: str) -> List[Dict]:
        """Process every frame of the video"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_count / fps
            detections = self.process_frame(frame)
            
            if detections:
                # Match with CLIP
                matched_detections = self.clip.match_detections(
                    frame,
                    detections,
                    self.current_prompt,
                    similarity_threshold=self.config['clip']['similarity_threshold']
                )
                
                if matched_detections:
                    results.append({
                        'timestamp': timestamp,
                        'detections': matched_detections
                    })

            frame_count += 1

        cap.release()
        return results

    def process_video_frame_skip(self, video_path: str) -> List[Dict]:
        """Process every N-th frame of the video"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        results = []
        skip_frames = self.config['modes']['frame_skip']['skip_frames']

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:
                timestamp = frame_count / fps
                detections = self.process_frame(frame)
                
                if detections:
                    matched_detections = self.clip.match_detections(
                        frame,
                        detections,
                        self.current_prompt,
                        similarity_threshold=self.config['clip']['similarity_threshold']
                    )
                    
                    if matched_detections:
                        results.append({
                            'timestamp': timestamp,
                            'detections': matched_detections
                        })

            frame_count += 1

        cap.release()
        return results

    def process_video_motion_filter(self, video_path: str) -> List[Dict]:
        """Process only frames with motion"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if self.detect_motion(frame):
                timestamp = frame_count / fps
                detections = self.process_frame(frame)
                
                if detections:
                    matched_detections = self.clip.match_detections(
                        frame,
                        detections,
                        self.current_prompt,
                        similarity_threshold=self.config['clip']['similarity_threshold']
                    )
                    
                    if matched_detections:
                        results.append({
                            'timestamp': timestamp,
                            'detections': matched_detections
                        })

            frame_count += 1

        cap.release()
        return results

    def process_video_track_then_match(self, video_path: str) -> List[Dict]:
        """Track objects across frames and match once"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        results = []
        keyframe_interval = self.config['modes']['track_then_match']['keyframe_interval']
        tracked_objects = {}  # Store tracked objects and their CLIP matches

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_count / fps

            if frame_count % keyframe_interval == 0:
                # Detect objects in keyframe
                detections = self.process_frame(frame)
                
                if detections:
                    # Convert detections to DeepSORT format
                    bboxes = []
                    confidences = []
                    class_ids = []
                    
                    for det in detections:
                        bbox = det['bbox']
                        bboxes.append([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]])
                        confidences.append(det['confidence'])
                        class_ids.append(det['class'])
                    
                    # Update tracker
                    tracks = self.tracker.update_tracks(
                        bboxes,
                        confidences,
                        class_ids,
                        frame
                    )
                    
                    # Match new tracks with CLIP
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                            
                        track_id = track.track_id
                        if track_id not in tracked_objects:
                            # Get bounding box
                            ltrb = track.to_ltrb()
                            x1, y1, x2, y2 = map(int, ltrb)
                            
                            # Extract region and match with CLIP
                            region = frame[y1:y2, x1:x2]
                            if region.size > 0:  # Ensure region is valid
                                similarity = self.clip.match_image(
                                    region,
                                    self.current_prompt
                                )
                                
                                if similarity > self.config['clip']['similarity_threshold']:
                                    tracked_objects[track_id] = {
                                        'first_seen': timestamp,
                                        'last_seen': timestamp,
                                        'similarity': similarity,
                                        'bbox': [x1, y1, x2, y2]
                                    }
                        else:
                            # Update last seen time
                            tracked_objects[track_id]['last_seen'] = timestamp
                            tracked_objects[track_id]['bbox'] = track.to_ltrb()

            frame_count += 1

        # Convert tracked objects to results
        for track_id, obj in tracked_objects.items():
            results.append({
                'timestamp': obj['first_seen'],
                'detections': [{
                    'bbox': obj['bbox'],
                    'confidence': obj['similarity'],
                    'track_id': track_id,
                    'duration': obj['last_seen'] - obj['first_seen']
                }]
            })

        cap.release()
        return results

    def process_video(self, video_path: str, prompt: str, mode: str = "full_scan") -> List[Dict]:
        """Process video using the specified mode"""
        self.current_prompt = prompt
        
        if mode == "full_scan":
            return self.process_video_full_scan(video_path)
        elif mode == "frame_skip":
            return self.process_video_frame_skip(video_path)
        elif mode == "motion_filter":
            return self.process_video_motion_filter(video_path)
        elif mode == "track_then_match":
            return self.process_video_track_then_match(video_path)
        else:
            raise ValueError(f"Unknown mode: {mode}") 