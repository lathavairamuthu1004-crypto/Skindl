import React, { useState, useEffect } from 'react';
import { Search, Calendar, Users, FileText, Settings, Bell, MoreHorizontal } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import API_URL from '../config';
import '../styles/DoctorDashboard.css';

const DoctorDashboard = () => {
  const { user, logout, onlineUsers } = useAuth();
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const response = await fetch(`${API_URL}/api/patients`);
        const data = await response.json();
        // Since database only holds name/email, we'll mock the condition/threat for now
        const enrichedData = data.map(p => ({
          ...p,
          id: p._id,
          condition: "Not Analyzed",
          threat: "Low",
          date: new Date(p.createdAt || Date.now()).toLocaleDateString(),
          status: "New"
        }));
        setPatients(enrichedData);
        setLoading(false);
      } catch (err) {
        console.error('Error:', err);
        setLoading(false);
      }
    };
    fetchPatients();
  }, []);

  return (
    <div className="doctor-layout">
      <aside className="sidebar glass">
        <div className="brand">
          <div className="logo-sq">SD</div>
          <span>SkinDL Pro</span>
        </div>

        <nav className="side-nav">
          <button className="nav-item active"><Users size={20} /> Patients</button>
          <button className="nav-item"><Calendar size={20} /> Schedule</button>
          <button className="nav-item"><FileText size={20} /> Lab Reports</button>
          <button className="nav-item"><Settings size={20} /> Settings</button>
        </nav>

        <div className="side-footer">
          <div className="mini-profile">
            <div className="avatar-sm">DS</div>
            <div className="profile-info">
              <strong>{user?.name || 'Dr. Smith'}</strong>
              <span>Dermatologist</span>
            </div>
          </div>
          <button className="logout-link" onClick={logout}>Sign Out</button>
        </div>
      </aside>

      <main className="doctor-content">
        <header className="content-header">
          <div className="search-bar glass">
            <Search size={18} color="var(--text-muted)" />
            <input type="text" placeholder="Search patients, conditions..." />
          </div>
          <div className="header-actions">
            <button className="icon-btn glass"><Bell size={20} /></button>
            <button className="btn-primary">New Appointment</button>
          </div>
        </header>

        <section className="stats-row">
          <div className="stat-card glass">
            <Users color="var(--primary)" />
            <div>
              <span className="label">Total Patients</span>
              <span className="value">{patients.length}</span>
            </div>
          </div>
          <div className="stat-card glass">
            <FileText color="var(--secondary)" />
            <div>
              <span className="label">Reports Pending</span>
              <span className="value">12</span>
            </div>
          </div>
          <div className="stat-card glass">
            <Calendar color="var(--accent)" />
            <div>
              <span className="label">Consultations Today</span>
              <span className="value">8</span>
            </div>
          </div>
        </section>

        <section className="online-patients-section">
          <div className="section-header">
            <h3>Online Patients ({onlineUsers.length})</h3>
            <p>Patients currently active in their portals.</p>
          </div>
          <div className="online-chips">
            {onlineUsers.map(u => (
              <div key={u.id} className="online-chip glass">
                <div className="status-dot"></div>
                <div className="chip-info">
                  <strong>{u.name}</strong>
                  <span>{u.lastActive}</span>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="patient-table-section glass">
          <div className="table-header">
            <h2>Recent Consultations</h2>
            <button className="view-all">View All</button>
          </div>

          <table className="patient-table">
            <thead>
              <tr>
                <th>Patient Name</th>
                <th>Condition</th>
                <th>Threat Level</th>
                <th>Scan Date</th>
                <th>Status</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan="6" style={{ textAlign: 'center', padding: '2rem' }}>Loading patients...</td></tr>
              ) : patients.length === 0 ? (
                <tr><td colSpan="6" style={{ textAlign: 'center', padding: '2rem' }}>No patients found.</td></tr>
              ) : patients.map(p => (
                <tr key={p.id}>
                  <td>
                    <div className="name-cell">
                      <div className="avatar-xs">{p.name.split(' ').map(n => n[0]).join('')}</div>
                      <div className="name-wrapper">
                        {p.name}
                        {p.online && <span className="online-indicator"></span>}
                      </div>
                    </div>
                  </td>
                  <td>{p.condition}</td>
                  <td>
                    <span className={`threat-tag ${p.threat.toLowerCase()}`}>
                      {p.threat}
                    </span>
                  </td>
                  <td>{p.date}</td>
                  <td>
                    <span className={`status-tag ${p.status.toLowerCase()}`}>
                      {p.status}
                    </span>
                  </td>
                  <td>
                    <button className="icon-btn-sm"><MoreHorizontal size={16} /></button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </main>
    </div>
  );
};

export default DoctorDashboard;
