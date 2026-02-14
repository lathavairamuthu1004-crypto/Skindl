const express = require('express');
const router = express.Router();
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const Patient = require('../models/Patient');
const Dermatologist = require('../models/Dermatologist');

// SIGNUP ROUTE
router.post('/signup', async (req, res) => {
    const { name, email, password, role } = req.body;

    try {
        const Model = role === 'doctor' ? Dermatologist : Patient;

        // Check if user exists
        let user = await Model.findOne({ email });
        if (user) return res.status(400).json({ message: 'User already exists' });

        // Hash password
        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(password, salt);

        // Create user
        user = new Model({
            name,
            email,
            password: hashedPassword,
            role
        });

        await user.save();
        res.status(201).json({ message: 'User registered successfully' });

    } catch (err) {
        console.error(err);
        res.status(500).json({ message: 'Server error' });
    }
});

// LOGIN ROUTE
router.post('/login', async (req, res) => {
    const { email, password, role } = req.body;

    try {
        const Model = role === 'doctor' ? Dermatologist : Patient;

        // Check if user exists
        const user = await Model.findOne({ email });
        if (!user) return res.status(400).json({ message: 'Invalid credentials' });

        // Validate password
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) return res.status(400).json({ message: 'Invalid credentials' });

        // Return user data (excluding password)
        const userData = {
            id: user._id,
            name: user.name,
            email: user.email,
            role: user.role
        };

        res.json({ user: userData });

    } catch (err) {
        console.error(err);
        res.status(500).json({ message: 'Server error' });
    }
});

module.exports = router;
