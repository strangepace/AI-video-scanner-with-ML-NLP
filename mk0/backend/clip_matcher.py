import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
import cv2
import time

class CLIPMatcher:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Cache for text encodings to avoid recomputing for the same prompt
        self.text_encoding_cache = {}

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text prompt using CLIP
        """
        # Check cache first
        if text in self.text_encoding_cache:
            return self.text_encoding_cache[text]
            
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        normalized_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Cache the result
        self.text_encoding_cache[text] = normalized_features
        
        return normalized_features

    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Encode image using CLIP
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        return image_features / image_features.norm(dim=-1, keepdim=True)

    def encode_image_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Encode a batch of images for better performance
        """
        # Convert BGR to RGB and to PIL
        pil_images = []
        for image in images:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(image))
        
        # Process batch of images
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        return image_features / image_features.norm(dim=-1, keepdim=True)

    def compute_similarity(self, text_features: torch.Tensor, image_features: torch.Tensor) -> float:
        """
        Compute similarity between text and image features
        """
        similarity = torch.matmul(text_features, image_features.T)
        return float(similarity[0][0])

    def compute_batch_similarity(self, text_features: torch.Tensor, image_features_batch: torch.Tensor) -> List[float]:
        """
        Compute similarity between text and a batch of image features
        """
        similarity = torch.matmul(text_features, image_features_batch.T)
        return similarity[0].cpu().numpy().tolist()

    def match_detections(self, frame: np.ndarray, detections: List[Dict], prompt: str, 
                        similarity_threshold: float = 0.25) -> List[Dict]:
        """
        Match detections in a frame with the text prompt
        """
        if not detections:
            return []
            
        text_features = self.encode_text(prompt)
        matched_detections = []

        # Extract all regions first
        regions = []
        valid_indices = []
        
        for i, detection in enumerate(detections):
            # Extract bounding box
            x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
            
            # Extract region of interest
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
                
            regions.append(roi)
            valid_indices.append(i)
        
        if not regions:
            return []
        
        # Batch encode all regions
        start_time = time.time()
        image_features_batch = self.encode_image_batch(regions)
        
        # Compute similarities for all regions at once
        similarities = self.compute_batch_similarity(text_features, image_features_batch)
        
        # Match detections based on similarities
        for idx, similarity in zip(valid_indices, similarities):
            if similarity > similarity_threshold:
                detection = detections[idx].copy()
                detection['similarity'] = similarity
                matched_detections.append(detection)

        return matched_detections

    def process_video_segment(self, video_path: str, start_time: float, end_time: float, 
                            prompt: str, sample_rate: int = 1) -> List[Dict]:
        """
        Process a segment of video and return matches
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        matches = []
        frame_count = start_frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        while frame_count <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                timestamp = frame_count / fps
                # Process frame and get matches
                # Note: This assumes detections are already available
                # You'll need to integrate this with the VideoDetector class
                matches.append({
                    'timestamp': timestamp,
                    'frame': frame
                })

            frame_count += 1

        cap.release()
        return matches 