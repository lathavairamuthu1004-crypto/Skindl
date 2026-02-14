const mongoose = require('mongoose');
const dotenv = require('dotenv');

dotenv.config({ path: './backend/.env' });

const seedDB = async () => {
    try {
        await mongoose.connect(process.env.MONGODB_URI);
        console.log('Connected to MongoDB...');

        // Create a temporary collection and insert one document
        const TestSchema = new mongoose.Schema({ name: String });
        const Test = mongoose.model('InitConnection', TestSchema);

        await Test.create({ name: 'Database Initialized' });

        console.log('✅ Success! Data saved. The "skindl" database should now be visible.');
        process.exit(0);
    } catch (err) {
        console.error('❌ Error:', err);
        process.exit(1);
    }
};

seedDB();
