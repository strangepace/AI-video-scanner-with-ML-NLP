# Setting up the Backend in Google Colab

1. Create a new Colab notebook
2. Install required packages:
```python
!pip install -r requirements.txt
```

3. Upload the backend code:
```python
# Upload colab_backend.py to Colab
from google.colab import files
uploaded = files.upload()
```

4. Set up ngrok authentication (if needed):
```python
!pip install pyngrok
from pyngrok import ngrok
# Set your authtoken if you have one
# ngrok.set_auth_token('your_token_here')
```

5. Run the backend:
```python
!python colab_backend.py
```

6. The backend will print a public URL that looks like `https://xxxx.ngrok.io`. Use this URL in your frontend by updating the `API_URL` in your frontend code.

7. To download logs after running:
```python
from google.colab import files
files.download('logs/backend.log')
```

## Troubleshooting

1. If you see CUDA/cuDNN warnings, these can be safely ignored as they don't affect functionality.

2. If you get a ModuleNotFoundError, make sure all requirements are installed:
```python
!pip install -r requirements.txt
```

3. If the backend seems stuck, you can stop it using the "Stop" button in Colab and restart it.

4. Memory issues:
   - Use Runtime > Factory reset runtime to clear all memory
   - Make sure you're using a GPU runtime (Runtime > Change runtime type > GPU)

5. For log access:
   - Stop the backend first
   - Then run the download command
   - Logs are in the `logs` directory 