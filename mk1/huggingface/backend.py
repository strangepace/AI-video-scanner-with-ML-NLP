from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64
import cv2
import numpy as np
import torch
from PIL import Image
import io
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CONFIG = {
    "yolo": {
        "model_path": "models/yolov8n.pt",
        "confidence_threshold": 0.5,
        "classes": [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
    },
    "clip": {
        "model_name": "openai/clip-vit-base-patch32",
        "similarity_threshold": 0.3
    }
}

# Force CPU usage
device = 'cpu'
torch.set_num_threads(4)  # Limit number of CPU threads

# Initialize models
yolo_model = None
clip_model = None
clip_processor = None

def initialize_models():
    global yolo_model, clip_model, clip_processor
    
    if yolo_model is None:
        print("Loading YOLO model...")
        yolo_model = YOLO(CONFIG["yolo"]["model_path"])
        yolo_model.to(device)
        # Use smaller batch size for CPU
        yolo_model.batch = 1
    
    if clip_model is None:
        print("Loading CLIP model...")
        clip_model = CLIPModel.from_pretrained(CONFIG["clip"]["model_name"])
        clip_processor = CLIPProcessor.from_pretrained(CONFIG["clip"]["model_name"])
        clip_model.to(device)
        # Use smaller batch size for CPU
        clip_model.config.max_position_embeddings = 77

@app.on_event("startup")
async def startup_event():
    initialize_models()

@app.post("/process_frame")
async def process_frame(request: dict):
    try:
        # Decode base64 frame
        frame_data = base64.b64decode(request["frame"])
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Resize frame to reduce processing time
        height, width = frame.shape[:2]
        max_dim = 640  # Limit maximum dimension
        if height > width:
            new_height = max_dim
            new_width = int(width * (max_dim / height))
        else:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Get prompt and mode
        prompt = request["prompt"]
        mode = request["mode"]
        
        # Process frame with YOLO
        results = yolo_model(frame, verbose=False)[0]
        matches = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            if conf < CONFIG["yolo"]["confidence_threshold"]:
                continue
                
            obj_img = frame[y1:y2, x1:x2]
            if obj_img.size == 0:
                continue
                
            # Convert to PIL Image
            obj_pil = Image.fromarray(cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB))
            
            # Process with CLIP
            inputs = clip_processor(
                text=[prompt],
                images=obj_pil,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = clip_model(**inputs)
                similarity = torch.sigmoid(outputs.logits_per_image).item()
            
            if similarity > CONFIG["clip"]["similarity_threshold"]:
                matches.append({
                    "class": results.names[cls],
                    "confidence": conf,
                    "similarity": similarity,
                    "bbox": [x1, y1, x2, y2]
                })
        
        return {"matches": matches}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 