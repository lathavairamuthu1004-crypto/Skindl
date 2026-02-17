import React, { createContext, useContext, useState } from 'react';
import API_URL from '../config';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [onlineUsers, setOnlineUsers] = useState([
    { id: 101, name: "Sarah Johnson", lastActive: "Just now" },
    { id: 102, name: "Marcus Chen", lastActive: "2 mins ago" }
  ]);

  const login = async (role, email, password) => {
    try {
      const response = await fetch(`${API_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, role })
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.message || 'Login failed');

      setUser(data.user);
      return { success: true };
    } catch (err) {
      return { success: false, message: err.message };
    }
  };

  const signup = async (role, name, email, password) => {
    try {
      const response = await fetch(`${API_URL}/api/auth/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, password, role })
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.message || 'Signup failed');

      return { success: true };
    } catch (err) {
      return { success: false, message: err.message };
    }
  };

  const logout = () => {
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, signup, logout, onlineUsers }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
