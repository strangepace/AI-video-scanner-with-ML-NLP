import subprocess
import time
from pyngrok import ngrok
import uvicorn
import nest_asyncio
import sys
import os

def setup_and_start():
    # Install required packages if not already installed
    subprocess.run([
        "pip", "install", 
        "fastapi", "uvicorn", "pyngrok", "ultralytics", 
        "transformers", "torch", "torchvision", 
        "opencv-python-headless", "pillow", "nest-asyncio"
    ], check=True)

    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("uploads/previews", exist_ok=True)

    # Start ngrok tunnel
    port = 8000
    try:
        # Kill any existing ngrok processes
        ngrok.kill()
        
        # Start new tunnel
        tunnel = ngrok.connect(port)
        public_url = tunnel.public_url
        
        print("\n" + "="*50)
        print(f"Public URL for frontend: {public_url}")
        print("="*50 + "\n")
        
        # Start the FastAPI server
        uvicorn.run(
            "colab_backend:app",
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        ngrok.kill()

if __name__ == "__main__":
    # Apply nest_asyncio
    nest_asyncio.apply()
    
    # Run setup and start server
    setup_and_start() 