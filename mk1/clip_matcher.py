import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import List, Dict
import cv2
from PIL import Image

class CLIPMatcher:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def match_detections(self, frame: np.ndarray, detections: List[Dict], prompt: str, 
                        similarity_threshold: float = 0.2) -> List[Dict]:
        """Match detections with the prompt using CLIP"""
        if not detections:
            return []
        
        # Convert frame to PIL Image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Extract regions for each detection
        regions = []
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            region = frame_pil.crop((x1, y1, x2, y2))
            regions.append(region)
        
        # Process images and text
        inputs = self.processor(
            text=[prompt],
            images=regions,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
            
            # Calculate similarities
            similarities = (image_embeds @ text_embeds.T).squeeze()
        
        # Filter and return matches
        matches = []
        for i, (det, similarity) in enumerate(zip(detections, similarities)):
            if similarity > similarity_threshold:
                matches.append({
                    **det,
                    'similarity': float(similarity)
                })
        
        return matches
    
    def match_image(self, image: np.ndarray, prompt: str) -> float:
        """Match a single image with the prompt"""
        # Convert to PIL Image
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Process image and text
        inputs = self.processor(
            text=[prompt],
            images=[image_pil],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
            
            # Calculate similarity
            similarity = (image_embeds @ text_embeds.T).squeeze()
        
        return float(similarity) 