const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const path = require('path');
const { exec } = require('child_process');

const app = express();
const PORT = 3000;
const FLASK_URL = 'http://localhost:5000';

// ── Middleware ────────────────────────────────────────
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

// ── Home Route ────────────────────────────────────────
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ── Predict Route ─────────────────────────────────────
app.post('/predict', async (req, res) => {
    try {
        const response = await axios.post(`${FLASK_URL}/predict`, req.body);
        res.json(response.data);
    } catch (error) {
        res.json({ error: 'Flask API not reachable. Make sure it is running!' });
    }
});

// ── Inventory Route ───────────────────────────────────
app.get('/inventory', async (req, res) => {
    try {
        const city = req.query.city || '';
        const response = await axios.get(`${FLASK_URL}/inventory?city=${city}`);
        res.json(response.data);
    } catch (error) {
        res.json({ error: 'Flask API not reachable!' });
    }
});

// ── Clusters Route ────────────────────────────────────
app.get('/clusters', async (req, res) => {
    try {
        const response = await axios.get(`${FLASK_URL}/clusters`);
        res.json(response.data);
    } catch (error) {
        res.json({ error: 'Flask API not reachable!' });
    }
});

// ── Open Power BI Route ───────────────────────────────
app.get('/open-powerbi', (req, res) => {
    const filePath = 'C:\\Users\\Humaira\\Desktop\\PROJECT(BI).pbix';
    exec(`start "" "${filePath}"`, (error) => {
        if (error) {
            console.error('❌ Error opening Power BI:', error.message);
            res.json({ success: false, error: error.message });
        } else {
            console.log('✅ Power BI file opened successfully!');
            res.json({ success: true });
        }
    });
});

// ── Start Server ──────────────────────────────────────
app.listen(PORT, () => {
    console.log(`✅ Node.js server running on http://localhost:${PORT}`);
});