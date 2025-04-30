import React, { useState } from 'react';
import axios from 'axios';
import {
  Box,
  Button,
  TextField,
  Typography,
  Card,
  CardContent,
  Grid,
  CircularProgress,
  Alert,
  Container,
  CssBaseline,
  ThemeProvider,
  createTheme
} from '@mui/material';

const API_URL = 'https://a637-34-169-58-32.ngrok-free.app';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    background: {
      default: '#f4f6fa',
    },
  },
});

function formatTime(seconds) {
  const min = Math.floor(seconds / 60);
  const sec = Math.floor(seconds % 60);
  return `${min}:${sec.toString().padStart(2, '0')}`;
}

function App() {
  const [file, setFile] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadedFilename, setUploadedFilename] = useState(null);

  const handleFileChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      setFile(event.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a video file');
      return;
    }
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await axios.post(`${API_URL}/upload`, formData);
      setUploadedFilename(response.data.file_id || response.data.filename);
      setError(null);
    } catch (err) {
      setError('Error uploading video. Please try again.');
      console.error('Upload error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!uploadedFilename) {
      setError('Please upload a video first');
      return;
    }
    if (!searchQuery.trim()) {
      setError('Please enter a search query');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/search`, null, {
        params: {
          file_id: uploadedFilename,
          prompt: searchQuery,
        },
      });
      if (!response.data.ok) {
        const error = response.data.error;
        setError(error.detail);
      } else {
        setResults(response.data.results || []);
      }
    } catch (error) {
      console.error('Search failed:', error);
      setError('Error searching video. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Poll status every few seconds
  const checkStatus = async () => {
    const response = await fetch(`${API_URL}/status`);
    const status = await response.json();
    if (status.active) {
      console.log(`Progress: ${status.progress}%, Segments found: ${status.segments_found}`);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Box sx={{ textAlign: 'center', mb: 4 }}>
          <Typography variant="h3" fontWeight={700} color="primary" gutterBottom>
            AI Video Search
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Upload a video and search for specific moments using natural language.
          </Typography>
        </Box>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        <Card sx={{ mb: 3, boxShadow: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              1. Upload Video
            </Typography>
            <input
              accept="video/*"
              type="file"
              onChange={handleFileChange}
              style={{ marginBottom: 16, display: 'block' }}
            />
            <Button
              variant="contained"
              onClick={handleUpload}
              disabled={!file || loading}
              sx={{ minWidth: 120 }}
            >
              {loading ? <CircularProgress size={24} /> : 'Upload'}
            </Button>
          </CardContent>
        </Card>
        <Card sx={{ mb: 3, boxShadow: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              2. Search Video
            </Typography>
            <TextField
              fullWidth
              label="Search Query"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              margin="normal"
              disabled={!uploadedFilename || loading}
            />
            <Button
              variant="contained"
              onClick={handleSearch}
              disabled={!uploadedFilename || loading || !searchQuery.trim()}
              sx={{ mt: 2, minWidth: 120 }}
            >
              {loading ? <CircularProgress size={24} /> : 'Search'}
            </Button>
          </CardContent>
        </Card>
        {results.length > 0 && (
          <Card sx={{ boxShadow: 3, mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Results
              </Typography>
              <Grid container spacing={2}>
                {results.map((seg, index) => (
                  <Grid item xs={12} sm={6} md={4} key={index}>
                    <Card sx={{ height: '100%' }}>
                      <CardContent>
                        <Typography variant="subtitle2" color="primary">
                          {seg.class_name || '--'} (Track #{seg.track_id})
                        </Typography>
                        <Typography variant="body2">
                          <b>From:</b> {formatTime(seg.start_time)} &nbsp; <b>To:</b> {formatTime(seg.end_time)}
                        </Typography>
                        <Typography variant="body2">
                          <b>Confidence:</b> {seg.confidence ? (seg.confidence * 100).toFixed(1) : '--'}%
                        </Typography>
                        <Box sx={{ mt: 1, mb: 1 }}>
                          {seg.video_preview ? (
                            <video
                              src={`${API_URL}/${seg.video_preview}`}
                              controls
                              style={{ width: '100%', borderRadius: 8 }}
                            />
                          ) : seg.thumbnail ? (
                            <img
                              src={`${API_URL}/${seg.thumbnail}`}
                              alt="Preview"
                              style={{ width: '100%', borderRadius: 8 }}
                            />
                          ) : null}
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        )}
      </Container>
    </ThemeProvider>
  );
}

export default App;
