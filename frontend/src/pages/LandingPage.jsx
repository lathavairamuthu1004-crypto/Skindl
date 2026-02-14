import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Shield, Activity, User, UserCheck, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';
import '../styles/LandingPage.css';

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <div className="landing-container">
      <nav className="navbar glass animate-fade-in">
        <div className="logo">
          <Shield size={32} color="var(--primary)" />
          <span>SkinDL</span>
        </div>
        <div className="nav-links">
          <button className="btn-secondary" onClick={() => navigate('/login?role=patient')}>Patient Portal</button>
          <button className="btn-primary" onClick={() => navigate('/login?role=doctor')}>Dermatologist Portal</button>
        </div>
      </nav>

      <main className="hero">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="hero-content"
        >
          <h1 className="hero-title">AI-Powered <span className="gradient-text">Skin Health</span> Detection</h1>
          <p className="hero-subtitle">
            Empowering patients and dermatologists with state-of-the-art computer vision to identify potential skin conditions early and accurately.
          </p>

          <div className="cta-group">
            <button className="cta-btn patient" onClick={() => navigate('/login?role=patient')}>
              <User size={24} />
              <div className="btn-label">
                <span>I am a</span>
                <strong>Patient</strong>
              </div>
              <ArrowRight size={20} className="arrow" />
            </button>

            <button className="cta-btn doctor" onClick={() => navigate('/login?role=doctor')}>
              <UserCheck size={24} />
              <div className="btn-label">
                <span>I am a</span>
                <strong>Dermatologist</strong>
              </div>
              <ArrowRight size={20} className="arrow" />
            </button>
          </div>
        </motion.div>

        <div className="hero-visual">
          <div className="floating-card glass c1">
            <Shield size={40} color="var(--primary)" />
            <p>Secure Analysis</p>
          </div>
          <div className="floating-card glass c2">
            <Activity size={40} color="var(--secondary)" />
            <p>Real-time Detection</p>
          </div>
          <div className="hero-image-placeholder glass">
            <div className="ai-pulse"></div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default LandingPage;
