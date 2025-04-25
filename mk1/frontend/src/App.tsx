import React from 'react';
import { Container, CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import VideoSearch from './components/VideoSearch';

const theme = createTheme({
  palette: {
    mode: 'light',
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container>
        <VideoSearch />
      </Container>
    </ThemeProvider>
  );
}

export default App; 