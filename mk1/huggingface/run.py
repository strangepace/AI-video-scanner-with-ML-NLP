import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

def run_backend():
    print("Starting backend server...")
    subprocess.run([sys.executable, "backend.py"])

def run_frontend():
    print("Starting frontend server...")
    subprocess.run([sys.executable, "app.py"])

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    
    # Start backend in a separate thread
    backend_thread = Thread(target=run_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Wait for backend to start
    time.sleep(2)
    
    # Start frontend
    run_frontend() 