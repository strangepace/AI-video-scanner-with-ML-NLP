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
import gc

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
            "skip_frames": 15,
            "description": "Process only every N frames"
        }
    },
    "yolo": {
        "model_path": "yolov8n",  # Use direct model name instead of local file
        "confidence_threshold": 0.45,
        "classes": [0, 1, 2, 3, 5, 7]
    },
    "clip": {
        "model_name": "openai/clip-vit-base-patch32",
        "similarity_threshold": 0.25
    },
    "output": {
        "num_frames": 3,
        "max_results": 10
    }
}

# Initialize models with better memory management
print("Initializing models...")
device = 'cpu'
torch.set_num_threads(4)

# Global variables for models
yolo_model = None
clip_model = None
clip_processor = None

def initialize_models():
    """Initialize models with proper error handling and memory management"""
    global yolo_model, clip_model, clip_processor
    
    try:
        # Clear any existing models from memory
        if yolo_model is not None:
            del yolo_model
        if clip_model is not None:
            del clip_model
        if clip_processor is not None:
            del clip_processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        # Initialize YOLO
        print("Loading YOLO model...")
        yolo_model = YOLO(CONFIG["yolo"]["model_path"])
        yolo_model.to(device)
        
        # Initialize CLIP with fast processing
        print("Loading CLIP model...")
        clip_model = CLIPModel.from_pretrained(
            CONFIG["clip"]["model_name"],
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        clip_processor = CLIPProcessor.from_pretrained(
            CONFIG["clip"]["model_name"],
            use_fast=True  # Enable fast processing
        )
        clip_model.to(device)
        
        print("Models loaded successfully")
        return True
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False

# Initialize models on startup
if not initialize_models():
    raise RuntimeError("Failed to initialize models")

def get_frames_around_timestamp(video_path: str, timestamp: float, num_frames: int) -> List[np.ndarray]:
    """Get frames around a specific timestamp"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        center_frame = int(timestamp * fps)
        half_window = num_frames // 2
        start_frame = max(0, center_frame - half_window)
        end_frame = min(total_frames, start_frame + num_frames)
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            # Resize frame for efficiency
            frame = cv2.resize(frame, (640, 360))
            frames.append(frame)
        
        cap.release()
        return frames
    except Exception as e:
        print(f"Error in get_frames_around_timestamp: {e}")
        return []

def frames_to_base64(frames: List[np.ndarray]) -> List[str]:
    """Convert frames to base64 strings"""
    base64_frames = []
    try:
        for frame in frames:
            # Compress image for faster transfer
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            base64_frames.append(base64.b64encode(buffer).decode('utf-8'))
        return base64_frames
    except Exception as e:
        print(f"Error in frames_to_base64: {e}")
        return []

def process_frame(frame: np.ndarray, prompt: str) -> List[Dict]:
    """Process a single frame with YOLO and CLIP"""
    try:
        # Resize frame to reduce processing time
        frame = cv2.resize(frame, (640, 360))
        
        # Process with YOLO
        results = yolo_model(frame, verbose=False)[0]
        matches = []
        
        for box in results.boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if conf < CONFIG["yolo"]["confidence_threshold"]:
                    continue
                    
                obj_img = frame[y1:y2, x1:x2]
                if obj_img.size == 0:
                    continue
                
                # Convert to PIL Image and ensure RGB
                obj_pil = Image.fromarray(cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB))
                
                # Process with CLIP using batched processing
                inputs = clip_processor(
                    text=[prompt],
                    images=obj_pil,
                    return_tensors="pt",
                    padding=True
                )
                
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
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
                
                # Clear some memory
                del inputs, outputs
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
                
            except Exception as box_error:
                print(f"Error processing box: {box_error}")
                continue
        
        return matches
    except Exception as e:
        print(f"Error in process_frame: {e}")
        return []

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
                try:
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
                        
                        if detection_frames:
                            results.append({
                                "timestamp": timestamp,
                                "matches": matches,
                                "frames": frames_to_base64(detection_frames)
                            })
                    
                    processed_frames += 1
                    # Update progress
                    if processed_frames % 5 == 0:  # Update every 5 frames
                        progress(min(100, (processed_frames / total_frames) * 100))
                except Exception as frame_error:
                    print(f"Error processing frame {frame_count}: {frame_error}")
                    continue
            
            frame_count += 1
            
        cap.release()
        
        if not results:
            return {"error": None, "results": []}
            
        # Sort by confidence and limit results
        results.sort(key=lambda x: max(m["similarity"] for m in x["matches"]), reverse=True)
        results = results[:CONFIG["output"]["max_results"]]
        
        return {"results": results, "error": None}
        
    except Exception as e:
        print(f"Error in process_video: {e}")
        return {"error": str(e), "results": []}

def create_results_html(results: List[Dict]) -> str:
    """Create HTML for displaying results with improved navigation and preview"""
    if not results:
        return "<div>No matches found</div>"
    
    html = """
    <div class='results-container'>
        <div class='results-summary'>
            Found {total_matches} matches in {num_scenes} scenes
        </div>
        <div class='results-navigation'>
            <button onclick='document.querySelector(".results-list").scrollTo(0, 0)' class='nav-button'>
                Top
            </button>
        </div>
    """.format(
        total_matches=sum(len(r["matches"]) for r in results),
        num_scenes=len(results)
    )
    
    html += "<div class='results-list'>"
    for idx, result in enumerate(results):
        timestamp = time.strftime('%H:%M:%S', time.gmtime(result["timestamp"]))
        frames = result["frames"]
        
        # Get highest confidence match
        best_match = max(result["matches"], key=lambda x: x["similarity"])
        confidence = f"{best_match['similarity']:.2%}"
        
        html += f"""
        <div class='result-item' id='result-{idx}'>
            <div class='result-header'>
                <div class='header-left'>
                    <span class='timestamp'>Timestamp: {timestamp}</span>
                    <span class='confidence'>Confidence: {confidence}</span>
                </div>
                <div class='header-right'>
                    <span class='scene-number'>Scene {idx + 1}</span>
                </div>
            </div>
            <div class='frames-container'>
                {"".join([f'''
                    <div class='frame-wrapper'>
                        <img src="data:image/jpeg;base64,{frame}" class="frame-image"/>
                        <div class='frame-overlay'>Frame {i + 1}</div>
                    </div>
                ''' for i, frame in enumerate(frames)])}
            </div>
            <div class='detections-container'>
                <div class='detections-header'>Detections:</div>
                <div class='detections-list'>
                    {"".join([f'''
                        <div class='detection-item'>
                            <span class='detection-class'>{m['class']}</span>
                            <span class='detection-confidence'>({m['similarity']:.2%})</span>
                        </div>
                    ''' for m in sorted(result["matches"], key=lambda x: x["similarity"], reverse=True)])}
                </div>
            </div>
        </div>
        """
    html += "</div></div>"
    return html

def analyze_video(video_input, prompt: str, mode: str, progress=gr.Progress()) -> tuple:
    """Main function to analyze video"""
    try:
        if video_input is None:
            return None, None, "Please upload a video file."
            
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
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, "Could not open video file"
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        cap.release()
        
        # Create video info HTML
        video_info = f"""
        <div class='video-info'>
            <div class='info-item'>Duration: {int(duration // 60)}m {int(duration % 60)}s</div>
            <div class='info-item'>Frames: {total_frames}</div>
            <div class='info-item'>FPS: {int(fps)}</div>
            <div class='info-item'>Mode: {mode}</div>
        </div>
        """
        
        # Process video
        results = process_video(video_path, prompt, mode, progress)
        
        # Clean up if we created a temporary file
        if isinstance(video_input, (bytes, tempfile._TemporaryFileWrapper)):
            os.unlink(video_path)
        
        if results.get("error"):
            return None, video_info, results["error"]
        
        # Create HTML results
        html_results = create_results_html(results["results"])
        return html_results, video_info, None
        
    except Exception as e:
        return None, None, str(e)

# Create Gradio interface
with gr.Blocks(css="""
    .video-info {
        display: flex;
        gap: 20px;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    .info-item {
        padding: 5px 10px;
        background: #e9ecef;
        border-radius: 4px;
        font-size: 0.9em;
        color: #495057;
    }
    .results-container {
        display: flex;
        flex-direction: column;
        gap: 20px;
        padding: 20px;
        max-height: 800px;
        overflow-y: auto;
    }
    .results-summary {
        position: sticky;
        top: 0;
        background: #fff;
        padding: 10px;
        border-bottom: 1px solid #dee2e6;
        z-index: 100;
        font-weight: bold;
    }
    .results-navigation {
        position: sticky;
        top: 40px;
        background: #fff;
        padding: 10px 0;
        z-index: 100;
    }
    .nav-button {
        padding: 5px 15px;
        background: #007bff;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .nav-button:hover {
        background: #0056b3;
    }
    .result-item {
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 8px;
        background: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    .header-left {
        display: flex;
        gap: 15px;
    }
    .timestamp {
        font-weight: bold;
        color: #2c3e50;
    }
    .confidence {
        color: #27ae60;
        font-weight: bold;
    }
    .scene-number {
        padding: 3px 8px;
        background: #6c757d;
        color: white;
        border-radius: 4px;
        font-size: 0.8em;
    }
    .frames-container {
        display: flex;
        gap: 15px;
        overflow-x: auto;
        padding: 10px 0;
        -webkit-overflow-scrolling: touch;
    }
    .frame-wrapper {
        position: relative;
        flex-shrink: 0;
    }
    .frame-image {
        height: 200px;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .frame-overlay {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(0,0,0,0.5);
        color: white;
        padding: 4px;
        font-size: 0.8em;
        text-align: center;
    }
    .detections-container {
        margin-top: 15px;
        padding-top: 10px;
        border-top: 1px solid #dee2e6;
    }
    .detections-header {
        font-weight: bold;
        margin-bottom: 8px;
        color: #343a40;
    }
    .detections-list {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .detection-item {
        background: #e9ecef;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.9em;
    }
    .detection-class {
        color: #495057;
    }
    .detection-confidence {
        color: #20c997;
        margin-left: 4px;
    }
    @media (max-width: 768px) {
        .header-left {
            flex-direction: column;
            gap: 5px;
        }
        .frame-image {
            height: 150px;
        }
    }
""") as demo:
    gr.Markdown("# AI Video Scanner with ML/NLP")
    gr.Markdown("Upload a video and enter a prompt to search for specific objects or scenes.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(
                label="Upload Video",
                format="mp4",
                height=300
            )
            prompt_input = gr.Textbox(
                label="Search Prompt",
                placeholder="Enter what you want to search for...",
                info="Be specific in your search (e.g., 'red car' instead of just 'car')"
            )
            mode_input = gr.Dropdown(
                choices=["full_scan", "frame_skip"],
                value="frame_skip",
                label="Processing Mode",
                info="Frame skip is faster but might miss quick events"
            )
            analyze_btn = gr.Button("Analyze Video", variant="primary")
        
        with gr.Column():
            video_info = gr.HTML(label="Video Information")
            gr.Markdown("### Analysis Results")
            results_output = gr.HTML()
            error_output = gr.Textbox(label="Error", visible=True)
    
    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input, prompt_input, mode_input],
        outputs=[results_output, video_info, error_output]
    )

if __name__ == "__main__":
    demo.launch() 