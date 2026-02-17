const express = require('express');
const router = express.Router();
const axios = require('axios');

// @route   POST /api/analyze
// @desc    Send image to Python AI service for CNN analysis
router.post('/', async (req, res) => {
    try {
        const { image } = req.body;

        if (!image) {
            return res.status(400).json({ message: 'No image data provided' });
        }

        // Call the Flask AI Service
        const aiServiceUrl = process.env.AI_SERVICE_URL || 'http://127.0.0.1:5002';
        const aiResponse = await axios.post(`${aiServiceUrl}/predict`, {
            image: image
        });

        res.json(aiResponse.data);
    } catch (err) {
        console.error('AI Service Error:', err.message);
        res.status(500).json({
            message: 'AI Service is unavailable. Make sure the Python Flask server is running on port 5002.',
            error: err.message
        });
    }
});

module.exports = router;
