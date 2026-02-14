import React, { useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { Shield, Mail, Lock, User, ArrowLeft } from 'lucide-react';
import { motion } from 'framer-motion';
import { useAuth } from '../context/AuthContext';
import '../styles/Login.css';

const Login = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { login, signup } = useAuth();
  const role = searchParams.get('role') || 'patient';
  const [isSignup, setIsSignup] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: ''
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    if (isSignup) {
      const res = await signup(role, formData.name, formData.email, formData.password);
      if (res.success) {
        setIsSignup(false); // Switch to login after successful signup
        alert('Signup successful! Please login.');
      } else {
        setError(res.message);
      }
    } else {
      const res = await login(role, formData.email, formData.password);
      if (res.success) {
        navigate(role === 'doctor' ? '/doctor-dashboard' : '/patient-dashboard');
      } else {
        setError(res.message);
      }
    }
    setLoading(false);
  };

  return (
    <div className="login-page">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="login-card glass"
      >
        <button className="back-btn" onClick={() => navigate('/')}>
          <ArrowLeft size={20} />
          Back
        </button>

        <div className="login-header">
          <div className="role-icon">
            {role === 'doctor' ? <Shield size={40} color="var(--secondary)" /> : <User size={40} color="var(--primary)" />}
          </div>
          <h1>{isSignup ? `Create ${role} Account` : `${role.charAt(0).toUpperCase() + role.slice(1)} Login`}</h1>
          <p>{isSignup ? 'Join us to get started.' : 'Welcome back! Please enter your details.'}</p>
        </div>

        {error && <div className="error-alert">{error}</div>}

        <form onSubmit={handleSubmit} className="login-form">
          {isSignup && (
            <div className="input-group">
              <label><User size={16} /> Full Name</label>
              <input
                type="text"
                placeholder="John Doe"
                required
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </div>
          )}

          <div className="input-group">
            <label><Mail size={16} /> Email Address</label>
            <input
              type="email"
              placeholder="name@example.com"
              required
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
            />
          </div>

          <div className="input-group">
            <label><Lock size={16} /> Password</label>
            <input
              type="password"
              placeholder="••••••••"
              required
              value={formData.password}
              onChange={(e) => setFormData({ ...formData, password: e.target.value })}
            />
          </div>

          <button type="submit" disabled={loading} className={`submit-btn ${role}`}>
            {loading ? 'Processing...' : (isSignup ? 'Create Account' : `Sign In as ${role.charAt(0).toUpperCase() + role.slice(1)}`)}
          </button>
        </form>

        <div className="login-footer">
          {isSignup ? 'Already have an account?' : "Don't have an account?"}
          <button className="toggle-mode" onClick={() => { setIsSignup(!isSignup); setError(''); }}>
            {isSignup ? ' Login here' : ' Sign up here'}
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default Login;
