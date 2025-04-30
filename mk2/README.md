# AI Video Scanner with ML & NLP (mk2)

## Project Overview
This version (mk2) focuses on running the backend on Google Colab for better GPU access and easier deployment. The project combines computer vision and natural language processing to enable natural language search within video content.

## Key Features
- Backend running on Google Colab with GPU support
- FastAPI server with ngrok tunneling
- React frontend with Material-UI
- YOLOv8 for object detection
- CLIP for text-image matching
- Real-time video processing

## Directory Structure
```
mk2/
├── backend/          # FastAPI backend optimized for Colab
├── frontend/         # React frontend
└── docs/            # Documentation and setup guides
```

## Setup Instructions

### Backend (Google Colab)
1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Change runtime type to GPU
4. Follow the instructions in `docs/colab_setup.md`

### Frontend (Local)
1. Navigate to frontend directory
2. Install dependencies: `npm install`
3. Start development server: `npm start`

## Development Workflow
1. Backend development happens in Colab
2. Frontend development happens locally
3. API communication via ngrok tunnel
4. Regular commits to track progress

## Dependencies
See `requirements.txt` for Python dependencies and `package.json` for frontend dependencies.

## Contributing
1. Create a new branch for features
2. Follow the established code style
3. Update documentation as needed
4. Submit pull requests for review 