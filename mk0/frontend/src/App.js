import React, { useState } from 'react';
import { 
  Container, 
  Box, 
  Typography, 
  TextField, 
  Button, 
  CircularProgress,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Alert,
  LinearProgress
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import SearchIcon from '@mui/icons-material/Search';
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary';

const API_URL = 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'video/mp4': ['.mp4']
    },
    maxFiles: 1,
    onDrop: acceptedFiles => {
      setFile(acceptedFiles[0]);
      setError('');
      setUploadProgress(0);
    }
  });

  const handleSearch = async () => {
    if (!file || !prompt) {
      setError('Please upload a video and enter a search prompt');
      return;
    }

    setLoading(true);
    setError('');
    setResults([]);
    setUploadProgress(0);

    try {
      // Upload video
      const formData = new FormData();
      formData.append('file', file);
      
      const uploadResponse = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percentCompleted);
        }
      });

      const fileId = uploadResponse.data.file_id;

      // Search video
      const searchResponse = await axios.post(`${API_URL}/search`, {
        file_id: fileId,
        prompt: prompt
      });

      if (searchResponse.data.results) {
        setResults(searchResponse.data.results);
      } else {
        setError('No results found');
      }
    } catch (err) {
      console.error('Error:', err);
      // Fix: Convert error object to string
      const errorMessage = err.response?.data?.detail 
        ? (typeof err.response.data.detail === 'object' 
            ? JSON.stringify(err.response.data.detail) 
            : err.response.data.detail)
        : 'An error occurred while processing your request';
      setError(errorMessage);
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          AI Video Search
        </Typography>

        <Paper 
          {...getRootProps()} 
          sx={{ 
            p: 3, 
            mb: 3, 
            textAlign: 'center',
            backgroundColor: isDragActive ? '#f0f0f0' : 'white',
            cursor: 'pointer'
          }}
        >
          <input {...getInputProps()} />
          <VideoLibraryIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
          <Typography>
            {isDragActive
              ? "Drop the video here"
              : "Drag and drop a video file here, or click to select"}
          </Typography>
          {file && (
            <Typography variant="body2" sx={{ mt: 1 }}>
              Selected: {file.name}
            </Typography>
          )}
        </Paper>

        <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
          <TextField
            fullWidth
            label="Search Prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g., man wearing red shirt"
          />
          <Button
            variant="contained"
            onClick={handleSearch}
            disabled={loading || !file || !prompt}
            startIcon={loading ? <CircularProgress size={20} /> : <SearchIcon />}
          >
            Search
          </Button>
        </Box>

        {uploadProgress > 0 && uploadProgress < 100 && (
          <Box sx={{ width: '100%', mb: 2 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Uploading: {uploadProgress}%
            </Typography>
            <LinearProgress variant="determinate" value={uploadProgress} />
          </Box>
        )}

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {String(error)}
          </Alert>
        )}

        {results.length > 0 && (
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Results
            </Typography>
            <List>
              {results.map((result, index) => (
                <ListItem key={index}>
                  <ListItemAvatar>
                    <Avatar src={`data:image/jpeg;base64,${result.preview}`} />
                  </ListItemAvatar>
                  <ListItemText
                    primary={`Match found at ${result.timestamp}`}
                    secondary={`Confidence: ${(result.confidence * 100).toFixed(1)}%`}
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        )}
      </Box>
    </Container>
  );
}

export default App; 