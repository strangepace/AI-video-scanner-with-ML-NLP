import gradio as gr
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from deep_sort_realtime.deepsort_tracker import DeepSort
from skimage.metrics import structural_similarity as ssim
import os
import yaml
import time
from tqdm import tqdm

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Download YOLO model if not present
if not os.path.exists("yolov8n.pt"):
    import subprocess
    subprocess.run(["wget", "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"])

# Load configuration
config = {
    'general': {
        'model_dir': "models",
        'upload_dir': "uploads",
        'max_results': 10
    },
    'yolo': {
        'model_path': "yolov8n.pt",
        'confidence_threshold': 0.5,
        'classes': [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
    },
    'clip': {
        'similarity_threshold': 0.3,
        'model_name': "openai/clip-vit-base-patch32"
    },
    'motion': {
        'threshold': 0.95
    },
    'frame_skip': {
        'skip_frames': 5
    }
}

class VideoDetector:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize YOLO model
        self.yolo_model = YOLO(config['yolo']['model_path'])
        self.yolo_model.to(self.device)
        
        # Initialize CLIP model
        self.clip_model = CLIPModel.from_pretrained(config['clip']['model_name'])
        self.clip_processor = CLIPProcessor.from_pretrained(config['clip']['model_name'])
        self.clip_model.to(self.device)
        
        # Initialize DeepSORT tracker
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

    def process_frame(self, frame, prompt):
        results = self.yolo_model(frame, verbose=False)[0]
        matches = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            if conf < config['yolo']['confidence_threshold']:
                continue
                
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
            
            if similarity > config['clip']['similarity_threshold']:
                matches.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'similarity': similarity,
                    'class_id': cls,
                    'class_name': results.names[cls]
                })
        
        return matches

    def track_objects(self, frame):
        results = self.yolo_model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            if conf < config['yolo']['confidence_threshold']:
                continue
                
            detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))
        
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks

    def detect_motion(self, frame, prev_frame):
        if prev_frame is None:
            return True
            
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        score, _ = ssim(gray1, gray2, full=True)
        return score < config['motion']['threshold']

    def process_video(self, video_path, prompt, mode='full_scan'):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = config['frame_skip']['skip_frames']
        
        results = []
        prev_frame = None
        frame_count = 0
        
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                if mode == 'frame_skip' and frame_count % frame_skip != 0:
                    pbar.update(1)
                    continue
                    
                if mode == 'motion_filter':
                    if not self.detect_motion(frame, prev_frame):
                        prev_frame = frame.copy()
                        pbar.update(1)
                        continue
                
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
                            
                            if similarity > config['clip']['similarity_threshold']:
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

def create_html_results(results, video_path):
    html = "<div style='font-family: Arial, sans-serif;'>"
    html += "<h2>Search Results</h2>"
    
    if not results:
        html += "<p>No matches found.</p>"
        return html
    
    # Sort by similarity and limit results
    results.sort(key=lambda x: x['similarity'], reverse=True)
    results = results[:config['general']['max_results']]
    
    for i, result in enumerate(results, 1):
        html += f"<div style='margin-bottom: 20px; padding: 10px; border: 1px solid #ddd;'>"
        html += f"<h3>Match {i}</h3>"
        html += f"<p>Time: {result['timestamp']:.2f}s (Frame {result['frame']})</p>"
        html += f"<p>Similarity: {result['similarity']:.2f}</p>"
        if 'class_name' in result:
            html += f"<p>Class: {result['class_name']}</p>"
        html += "</div>"
    
    html += "</div>"
    return html

def analyze_video(video, prompt, mode):
    if video is None:
        return "Please upload a video file."
    
    # Save uploaded video
    video_path = os.path.join("uploads", "temp_video.mp4")
    video.save(video_path)
    
    # Initialize detector
    detector = VideoDetector()
    
    # Process video
    results = detector.process_video(video_path, prompt, mode)
    
    # Create HTML results
    html_results = create_html_results(results, video_path)
    
    return html_results

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎥 AI Video Search Tool")
    gr.Markdown("Upload a video and describe what you're looking for in natural language.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            prompt_input = gr.Textbox(label="Search Prompt", placeholder="e.g., 'man in red shirt', 'white truck'")
            mode_input = gr.Radio(
                choices=["full_scan", "frame_skip", "motion_filter", "track_then_match"],
                value="frame_skip",
                label="Detection Mode"
            )
            analyze_btn = gr.Button("Search Video")
        
        with gr.Column():
            results_output = gr.HTML(label="Results")
    
    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input, prompt_input, mode_input],
        outputs=results_output
    )

if __name__ == "__main__":
    demo.launch() 