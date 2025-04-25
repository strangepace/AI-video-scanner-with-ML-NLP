import gradio as gr
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import time
from typing import List, Dict
import tempfile
import base64
import io

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Configuration
CONFIG = {
    "modes": {
        "full_scan": {
            "enabled": True,
            "description": "Process every frame using YOLOv8 and CLIP"
        },
        "frame_skip": {
            "enabled": True,
            "skip_frames": 10,  # Increased skip frames for CPU
            "description": "Process only every N frames"
        }
    },
    "yolo": {
        "model_path": "models/yolov8n.pt",
        "confidence_threshold": 0.6,  # Increased confidence threshold
        "classes": [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
    },
    "clip": {
        "model_name": "openai/clip-vit-base-patch32",
        "similarity_threshold": 0.7  # Increased similarity threshold
    },
    "output": {
        "num_frames": 5,  # Number of frames to show
        "max_results": 10  # limit number of results
    }
}

# Initialize models
print("Initializing models...")
device = 'cpu'
torch.set_num_threads(4)  # Limit CPU threads

# Initialize YOLO
if not os.path.exists(CONFIG["yolo"]["model_path"]):
    print("Downloading YOLO model...")
    import subprocess
    subprocess.run([
        "wget",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "-O",
        CONFIG["yolo"]["model_path"]
    ])

yolo_model = YOLO(CONFIG["yolo"]["model_path"])
yolo_model.to(device)

# Initialize CLIP
clip_model = CLIPModel.from_pretrained(CONFIG["clip"]["model_name"])
clip_processor = CLIPProcessor.from_pretrained(CONFIG["clip"]["model_name"])
clip_model.to(device)

def get_frames_around_timestamp(video_path: str, timestamp: float, num_frames: int) -> List[np.ndarray]:
    """Get frames around a specific timestamp"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame numbers
    center_frame = int(timestamp * fps)
    half_window = num_frames // 2
    start_frame = max(0, center_frame - half_window)
    
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def frames_to_base64(frames: List[np.ndarray]) -> List[str]:
    """Convert frames to base64 strings"""
    base64_frames = []
    for frame in frames:
        _, buffer = cv2.imencode('.jpg', frame)
        base64_frames.append(base64.b64encode(buffer).decode('utf-8'))
    return base64_frames

def process_frame(frame: np.ndarray, prompt: str) -> List[Dict]:
    """Process a single frame with YOLO and CLIP"""
    # Resize frame to reduce processing time
    height, width = frame.shape[:2]
    max_dim = 640
    if height > width:
        new_height = max_dim
        new_width = int(width * (max_dim / height))
    else:
        new_width = max_dim
        new_height = int(height * (max_dim / width))
    
    frame = cv2.resize(frame, (new_width, new_height))
    
    # Process with YOLO
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
    
    return matches

def process_video(video_path: str, prompt: str, mode: str, progress=gr.Progress()) -> Dict:
    """Process video and return results"""
    try:
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file"}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate total frames to process
        if mode == "frame_skip":
            total_frames = total_frames // CONFIG["modes"]["frame_skip"]["skip_frames"]
        
        results = []
        frame_count = 0
        processed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame based on mode
            if mode == "full_scan" or frame_count % CONFIG["modes"]["frame_skip"]["skip_frames"] == 0:
                # Process frame
                matches = process_frame(frame, prompt)
                
                if matches:
                    timestamp = frame_count / fps
                    # Get frames around detection
                    detection_frames = get_frames_around_timestamp(
                        video_path,
                        timestamp,
                        CONFIG["output"]["num_frames"]
                    )
                    
                    results.append({
                        "timestamp": timestamp,
                        "matches": matches,
                        "frames": frames_to_base64(detection_frames)
                    })
                
                processed_frames += 1
                # Ensure progress is between 0 and 100
                progress(min(100, (processed_frames / total_frames) * 100))
            
            frame_count += 1
            
        cap.release()
        
        # Sort by confidence and limit results
        results.sort(key=lambda x: max(m["similarity"] for m in x["matches"]), reverse=True)
        results = results[:CONFIG["output"]["max_results"]]
        
        return {"results": results, "error": None}
        
    except Exception as e:
        return {"error": str(e), "results": []}

def create_results_html(results: List[Dict]) -> str:
    """Create HTML for displaying results"""
    if not results:
        return "<div>No matches found</div>"
    
    html = "<div class='results-container'>"
    for result in results:
        timestamp = time.strftime('%H:%M:%S', time.gmtime(result["timestamp"]))
        frames = result["frames"]
        
        # Get highest confidence match
        best_match = max(result["matches"], key=lambda x: x["similarity"])
        confidence = f"{best_match['similarity']:.2%}"
        
        html += f"""
        <div class='result-item'>
            <div class='result-header'>
                <span class='timestamp'>Timestamp: {timestamp}</span>
                <span class='confidence'>Confidence: {confidence}</span>
            </div>
            <div class='frames-container'>
                {"".join([f'<img src="data:image/jpeg;base64,{frame}" class="frame-image"/>' for frame in frames])}
            </div>
            <div class='matches'>
                Detected: {", ".join([f"{m['class']} ({m['similarity']:.2%})" for m in result["matches"]])}
            </div>
        </div>
        """
    html += "</div>"
    return html

def analyze_video(video_input, prompt: str, mode: str, progress=gr.Progress()) -> tuple:
    """Main function to analyze video"""
    try:
        if video_input is None:
            return None, "Please upload a video file."
            
        # Handle different types of video input
        if isinstance(video_input, str):
            video_path = video_input
        else:
            # Save video to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                if hasattr(video_input, 'name'):
                    # If it's a file-like object
                    with open(video_input.name, 'rb') as f:
                        tmp_file.write(f.read())
                else:
                    # If it's bytes
                    tmp_file.write(video_input)
                video_path = tmp_file.name
        
        # Process video
        results = process_video(video_path, prompt, mode, progress)
        
        # Clean up if we created a temporary file
        if isinstance(video_input, (bytes, tempfile._TemporaryFileWrapper)):
            os.unlink(video_path)
        
        if results.get("error"):
            return None, results["error"]
        
        # Create HTML results
        html_results = create_results_html(results["results"])
        return html_results, None
        
    except Exception as e:
        return None, str(e)

# Create Gradio interface
with gr.Blocks(css="""
    .results-container {
        display: flex;
        flex-direction: column;
        gap: 20px;
        padding: 20px;
    }
    .result-item {
        border: 1px solid #ccc;
        padding: 15px;
        border-radius: 8px;
        background: #f8f9fa;
    }
    .result-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .timestamp {
        font-weight: bold;
        color: #2c3e50;
    }
    .confidence {
        color: #27ae60;
        font-weight: bold;
    }
    .frames-container {
        display: flex;
        gap: 10px;
        overflow-x: auto;
        padding: 10px 0;
    }
    .frame-image {
        height: 200px;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .matches {
        color: #666;
        margin-top: 10px;
        font-size: 0.9em;
    }
""") as demo:
    gr.Markdown("# AI Video Scanner with ML/NLP")
    gr.Markdown("Upload a video and enter a prompt to search for specific objects or scenes.")
    gr.Markdown("Note: Processing may take longer on CPU. For best results, use shorter videos.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(
                label="Upload Video",
                format="mp4"
            )
            prompt_input = gr.Textbox(
                label="Search Prompt",
                placeholder="Enter what you want to search for...",
                info="Be specific in your search (e.g., 'red car' instead of just 'car')"
            )
            mode_input = gr.Dropdown(
                choices=["full_scan", "frame_skip"],
                value="frame_skip",  # Default to frame_skip for CPU
                label="Processing Mode",
                info="Frame skip is faster but might miss quick events"
            )
            analyze_btn = gr.Button("Analyze Video", variant="primary")
        
        with gr.Column():
            results_output = gr.HTML(label="Results")
            error_output = gr.Textbox(label="Error", visible=True)
    
    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input, prompt_input, mode_input],
        outputs=[results_output, error_output]
    )

if __name__ == "__main__":
    demo.launch() 