const express = require('express');
const router = express.Router();
const Patient = require('../models/Patient');

// @route   GET /api/patients
// @desc    Get all registered patients
// @access  Public (for now, in a real app this should be restricted to doctors)
router.get('/', async (req, res) => {
    try {
        const patients = await Patient.find().select('-password').sort({ createdAt: -1 });
        res.json(patients);
    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server Error');
    }
});

module.exports = router;
