from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
from pathlib import Path
import uuid
import logging
from typing import List, Dict, Any
import torch
import cv2
import numpy as np
import uvicorn
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch.nn.functional as F
from datetime import datetime, timedelta
import json
import gc
from pyngrok import ngrok
import nest_asyncio
import asyncio

# Enhanced logging setup
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "backend.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
PREVIEW_DIR = UPLOAD_DIR / "previews"
PREVIEW_DIR.mkdir(exist_ok=True)

# Rest of the file content...
# [Previous content of colab_backend.py] 