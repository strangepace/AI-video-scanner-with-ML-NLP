import gradio as gr
import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
import tempfile
import time
from typing import List, Dict
import base64
from PIL import Image
import io
import os
import subprocess

# Import our custom modules
from advanced_detector import AdvancedVideoDetector
from utils import format_timestamp, extract_frame_preview

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Download YOLOv8 model if not present
yolo_model_path = "models/yolov8n.pt"
if not os.path.exists(yolo_model_path):
    print("Downloading YOLOv8 model...")
    subprocess.run(["wget", "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt", "-O", yolo_model_path])

# Load configuration
config = {
    "modes": {
        "full_scan": {
            "enabled": True,
            "description": "Process every frame using YOLOv8 and CLIP"
        },
        "frame_skip": {
            "enabled": True,
            "skip_frames": 5,
            "description": "Process only every N frames"
        },
        "motion_filter": {
            "enabled": True,
            "motion_threshold": 30,
            "min_contour_area": 500,
            "description": "Only analyze frames where motion is detected"
        },
        "track_then_match": {
            "enabled": True,
            "keyframe_interval": 30,
            "max_age": 30,
            "min_hits": 3,
            "description": "Track objects across frames and match once"
        }
    },
    "yolo": {
        "model": yolo_model_path,
        "confidence_threshold": 0.3,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
    "clip": {
        "similarity_threshold": 0.2,
        "batch_size": 4
    },
    "general": {
        "max_results": 10,
        "preview_size": [320, 240],
        "upload_dir": "uploads"
    }
}

# Initialize detector
video_detector = AdvancedVideoDetector(config)

def process_video(video_path: str, prompt: str, mode: str, progress=gr.Progress()) -> Dict:
    """
    Process video and return results
    """
    try:
        # Process video with selected mode
        results = video_detector.process_video(video_path, prompt, mode)
        
        # Format results
        formatted_results = []
        for result in results:
            timestamp = result['timestamp']
            for detection in result['detections']:
                # Extract relevant region from the frame
                frame = video_detector.get_frame_at_timestamp(video_path, timestamp)
                x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
                region = frame[y1:y2, x1:x2]
                
                # Create a preview
                preview_bytes = extract_frame_preview(frame)
                preview_base64 = base64.b64encode(preview_bytes).decode('utf-8')
                
                formatted_results.append({
                    "timestamp": format_timestamp(timestamp),
                    "confidence": detection.get('confidence', 0),
                    "preview": preview_base64,
                    "object_class": detection.get('class', 'unknown'),
                    "track_id": detection.get('track_id'),
                    "duration": detection.get('duration', 0)
                })
        
        # Sort by confidence and limit results
        formatted_results.sort(key=lambda x: x["confidence"], reverse=True)
        formatted_results = formatted_results[:config['general']['max_results']]
        
        if not formatted_results:
            return {
                "message": "No matches found for the given prompt",
                "results": []
            }
        
        return {
            "message": f"Found {len(formatted_results)} matches",
            "results": formatted_results
        }
    except Exception as e:
        return {
            "message": f"Error processing video: {str(e)}",
            "results": []
        }

def create_results_html(results: List[Dict]) -> str:
    """Create HTML to display results"""
    if not results:
        return "<p>No matches found</p>"
    
    html = "<div style='display: flex; flex-direction: column; gap: 1rem;'>"
    for result in results:
        html += f"""
        <div style='display: flex; gap: 1rem; align-items: center; padding: 1rem; border: 1px solid #ccc; border-radius: 4px;'>
            <img src='data:image/jpeg;base64,{result["preview"]}' style='width: 160px; height: 120px; object-fit: cover;' />
            <div>
                <p><strong>Time:</strong> {result["timestamp"]}</p>
                <p><strong>Confidence:</strong> {result["confidence"]:.2%}</p>
                <p><strong>Class:</strong> {result["object_class"]}</p>
                {f'<p><strong>Duration:</strong> {result["duration"]:.1f}s</p>' if result.get("duration") else ''}
            </div>
        </div>
        """
    html += "</div>"
    return html

def analyze_video(video: gr.Video, prompt: str, mode: str, progress=gr.Progress()) -> Dict:
    """
    Main function to analyze video
    """
    if video is None:
        return {"message": "Please upload a video", "results_html": ""}
    
    if not prompt:
        return {"message": "Please enter a search prompt", "results_html": ""}
    
    # Save video to temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_path = temp_file.name
        video.save(temp_path)
    
    try:
        # Process video
        result = process_video(temp_path, prompt, mode, progress)
        
        # Create HTML for results
        html = create_results_html(result["results"])
        
        return {
            "message": result["message"],
            "results_html": html
        }
    finally:
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)

# Create Gradio interface
with gr.Blocks(title="AI Video Search Tool") as demo:
    gr.Markdown("""
    # 🎥 AI Video Search Tool
    
    Upload a video and describe what you're looking for. The tool will analyze the video and find matching objects/people.
    """)
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video (MP4)")
            prompt_input = gr.Textbox(
                label="Search Prompt",
                placeholder="e.g., 'man in red shirt', 'white truck', 'elderly woman walking'"
            )
            mode_input = gr.Dropdown(
                choices=["full_scan", "frame_skip", "motion_filter", "track_then_match"],
                value="full_scan",
                label="Detection Mode"
            )
            submit_btn = gr.Button("Search Video")
        
        with gr.Column():
            message_output = gr.Textbox(label="Status")
            results_output = gr.HTML(label="Results")
    
    # Add mode descriptions
    gr.Markdown("""
    ### Detection Modes:
    - **Full Scan**: Process every frame (most accurate but slowest)
    - **Frame Skip**: Process every N frames (faster, configurable in config.yaml)
    - **Motion Filter**: Only process frames with motion (good for static cameras)
    - **Track Then Match**: Track objects across frames (efficient for moving objects)
    """)
    
    # Set up event handlers
    submit_btn.click(
        fn=analyze_video,
        inputs=[video_input, prompt_input, mode_input],
        outputs=[message_output, results_output]
    )

if __name__ == "__main__":
    demo.launch(share=True) 