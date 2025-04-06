# AI Video Scanner with ML & NLP

An intelligent video search system that allows users to find specific moments in videos using natural language queries. The system combines computer vision and natural language processing to understand and match user queries with video content.

## Project Overview

This project uses state-of-the-art ML models to enable natural language search within video content:
- **YOLOv8** for object detection
- **CLIP** for matching text descriptions with visual content
- **FastAPI** for the backend server
- **React** with Material-UI for the frontend interface

## Features (mk0)

- 🎥 Video upload and processing
- 🔍 Natural language search queries
- 🤖 Real-time object detection
- 📊 Confidence scoring for matches
- 🖼️ Preview thumbnails for matched moments
- ⚡ Optimized batch processing

## Directory Structure

```
mk0/
├── backend/
│   ├── main.py           # FastAPI server implementation
│   ├── detector.py       # YOLOv8 object detection
│   ├── clip_matcher.py   # CLIP text-image matching
│   ├── utils.py          # Helper functions
│   ├── requirements.txt  # Python dependencies
│   └── uploads/          # Video storage directory
│
├── frontend/
│   ├── src/
│   │   └── App.js       # Main React component
│   ├── public/
│   │   └── index.html
│   └── package.json     # Node.js dependencies
```

## Setup Instructions

### Backend Setup
```bash
cd mk0/backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

### Frontend Setup
```bash
cd mk0/frontend
npm install
npm start
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Usage

1. Open the web interface at http://localhost:3000
2. Upload a video using the drag-and-drop interface
3. Enter a natural language search query (e.g., "person in red shirt", "car on street")
4. View results with timestamps and confidence scores

## Technical Details

### Backend Components
- **FastAPI Server**: Handles video uploads and search requests
- **YOLOv8 Detector**: Processes video frames for object detection
- **CLIP Matcher**: Matches natural language queries with detected objects
- **Batch Processing**: Optimized frame analysis for better performance

### Frontend Components
- **React UI**: Modern, responsive interface
- **Material-UI**: Polished component library
- **File Upload**: Drag-and-drop video upload
- **Results Display**: Organized display of matches with previews

## Performance Optimizations

- Adaptive frame sampling based on video length
- Batch processing for efficient GPU utilization
- Text encoding caching for repeated queries
- Preview image optimization

## Future Improvements (Planned for mk1)
- [ ] Enhanced search accuracy
- [ ] Video preview player
- [ ] Progress indicators
- [ ] Advanced query support
- [ ] Results filtering
- [ ] Performance optimizations

## License

MIT License 