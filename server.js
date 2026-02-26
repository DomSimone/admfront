const express = require('express');
const path = require('path');
const axios = require('axios'); // Import axios
const app = express();

// Middleware to parse incoming JSON bodies
app.use(express.json());

// Serve static files from the current directory
// It's better practice to have a dedicated 'public' folder for your UI files
app.use(express.static(path.join(__dirname, 'public')));

// API endpoint to proxy requests to the backend
app.post('/api/prompt', async (req, res) => {
  try {
    const { prompt } = req.body;
    const backendUrl = 'https://admibckend-1.onrender.com';

    // Forward the prompt to the backend server
    // NOTE: This assumes the backend expects a POST request to its root ('/')
    // with a JSON body like { "prompt": "your message" }
    const backendResponse = await axios.post(backendUrl, { prompt });

    // Send the response from the backend back to the frontend client
    res.json(backendResponse.data);
  } catch (error) {
    console.error('Error proxying to backend:', error.message);
    res.status(500).json({ error: 'Failed to communicate with the backend server.' });
  }
});

// Serve index.html for all routes (SPA support)
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Frontend server running on port ${PORT}`);
});
