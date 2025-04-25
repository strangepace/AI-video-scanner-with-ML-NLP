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
} from '@mui/material';

const API_URL = 'http://localhost:8000';

interface SearchResult {
  timestamp: number;
  confidence: number;
  class_name: string;
  frame_path: string;
  bbox: number[];
}

const VideoSearch: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadedFilename, setUploadedFilename] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
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
      const response = await axios.post(`${API_URL}/upload-video`, formData);
      setUploadedFilename(response.data.filename);
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
      const response = await axios.post(`${API_URL}/search-video`, null, {
        params: {
          filename: uploadedFilename,
          query: searchQuery,
        },
      });
      setResults(response.data.results);
    } catch (err) {
      setError('Error searching video. Please try again.');
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ maxWidth: 800, margin: 'auto', padding: 3 }}>
      <Typography variant="h4" gutterBottom>
        Video Search
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Upload Video
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
          >
            {loading ? <CircularProgress size={24} /> : 'Upload'}
          </Button>
        </CardContent>
      </Card>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Search Video
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
            sx={{ mt: 2 }}
          >
            {loading ? <CircularProgress size={24} /> : 'Search'}
          </Button>
        </CardContent>
      </Card>

      {results.length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Results
            </Typography>
            <Grid container spacing={2}>
              {results.map((result, index) => (
                <Grid item xs={12} sm={6} md={4} key={index}>
                  <Card>
                    <CardContent>
                      <img
                        src={`${API_URL}/${result.frame_path}`}
                        alt={`Detection at ${result.timestamp}s`}
                        style={{ width: '100%', height: 'auto' }}
                      />
                      <Typography variant="body2">
                        Time: {result.timestamp.toFixed(2)}s
                      </Typography>
                      <Typography variant="body2">
                        Object: {result.class_name}
                      </Typography>
                      <Typography variant="body2">
                        Confidence: {(result.confidence * 100).toFixed(1)}%
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default VideoSearch; 