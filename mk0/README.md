# mk0 - Initial Implementation

This is the first version (mk0) of the AI Video Scanner project, implementing core functionality for video search using natural language.

## Implementation Details

### Backend Components

1. **FastAPI Server (`main.py`)**
   - Video upload endpoint
   - Search endpoint with natural language processing
   - Health check endpoint
   - CORS configuration for frontend access

2. **YOLO Detector (`detector.py`)**
   - YOLOv8 implementation for object detection
   - Frame extraction and processing
   - Batch processing for performance
   - Confidence-based filtering

3. **CLIP Matcher (`clip_matcher.py`)**
   - Text-to-image similarity matching
   - Natural language understanding
   - Batch processing of image regions
   - Caching for repeated queries

4. **Utilities (`utils.py`)**
   - Video frame extraction
   - Timestamp formatting
   - Preview image generation
   - Result merging and filtering

### Frontend Components

1. **Main App (`App.js`)**
   - Drag-and-drop video upload
   - Search interface
   - Results display with previews
   - Progress indicators

## Key Features

- Video upload with format validation
- Natural language search processing
- Real-time object detection
- Preview generation for matches
- Confidence scoring
- Timestamp-based results

## Performance Features

- Adaptive frame sampling
- Batch processing
- Text encoding cache
- Optimized preview generation

## Known Limitations

- Processing time increases with video length
- High resource usage during analysis
- Limited to MP4 video format
- Basic error handling

## Development Notes

- YOLOv8 nano model used for faster processing
- CLIP base model for text-image matching
- Material-UI components for frontend
- FastAPI for efficient API handling 