const http = require('http');
const https = require('https');
const { URL } = require('url');
const busboy = require('busboy');

// UPDATED: Point to your PythonAnywhere hosted service
const PYTHON_SERVICE_URL = 'https://domsimone.pythonanywhere.com/process';

const server = http.createServer((req, res) => {
    // CORS configuration for Vercel
    res.setHeader('Access-Control-Allow-Origin', 'https://admfront-ardc3akcb-pmpanashe489-3815s-projects.vercel.app');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    // UPDATED: Allow Vercel Frontend
    res.setHeader('Access-Control-Allow-Origin', 'https://admfront-k2li07hb0-pmpanashe489-3815s-projects.vercel.app');
    res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

    if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
    }

    if (reqUrl.pathname === '/api/documents/extract' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => body += chunk.toString());
        req.on('end', async () => {
            try {
                const { documentContent, prompt, filename } = JSON.parse(body);

                // Logic to forward to PythonAnywhere
                const formData = new FormData();
                const buffer = Buffer.from(documentContent, 'base64');
                formData.append('file', new Blob([buffer]), filename || 'doc.pdf');
                formData.append('prompt', prompt);

                const pythonRes = await fetch(PYTHON_SERVICE_URL, {
                    method: 'POST',
                    body: formData
                });
                const data = await pythonRes.json();

                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(data));
            } catch (e) {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: e.message }));
            }
        });
    } else {
        res.writeHead(404);
        res.end('Not Found');
    }
});

const PORT = process.env.PORT || 3001;
server.listen(3001);