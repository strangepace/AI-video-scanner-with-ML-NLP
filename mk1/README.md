# 🎥 AI Video Search Tool

A powerful video analysis tool that uses AI to search through videos using natural language prompts. Built with YOLOv8, CLIP, and Gradio.

## Features

- **Natural Language Search**: Describe what you're looking for in plain English
- **Multiple Detection Modes**:
  - **Full Scan**: Process every frame (most accurate)
  - **Frame Skip**: Process every N frames (faster)
  - **Motion Filter**: Only process frames with motion
  - **Track Then Match**: Track objects across frames
- **Real-time Processing**: Get results as the video is being analyzed
- **Visual Results**: See previews of matching frames with timestamps

## Usage

1. Upload a video file (MP4 format)
2. Enter your search prompt (e.g., "man in red shirt", "white truck", "elderly woman walking")
3. Select a detection mode based on your needs
4. Click "Search Video" and wait for results

## Technical Details

- **Object Detection**: YOLOv8 for fast and accurate object detection
- **Text-Image Matching**: CLIP for matching natural language prompts with visual content
- **Object Tracking**: DeepSORT for tracking objects across frames
- **Motion Detection**: OpenCV for efficient motion detection
- **GPU Acceleration**: Utilizes CUDA when available for faster processing

## Local Development

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```

## Deployment

This app is designed to run on Hugging Face Spaces with GPU acceleration. To deploy:

1. Create a new Space on Hugging Face
2. Upload all files from this repository
3. Enable GPU in the Space settings
4. The app will automatically install dependencies and download models

## Notes

- Processing time depends on video length and selected mode
- For best results, use clear and specific prompts
- The app supports various video formats but works best with MP4
- Results are limited to the top 10 matches by default

## License

MIT License 