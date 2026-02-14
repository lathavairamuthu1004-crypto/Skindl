const mongoose = require('mongoose');

const DermatologistSchema = new mongoose.Schema({
    name: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    role: { type: String, default: 'doctor' },
    specialization: { type: String, default: 'Medical Dermatology' },
    createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Dermatologist', DermatologistSchema);
