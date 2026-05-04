import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { Activity, Layers, Database, FileText, Settings, User } from 'lucide-react';

export default function Layout({ children }) {
  const location = useLocation();

  const menuItems = [
    { path: '/single', label: 'Tekli Tahmin', icon: <Activity size={20} /> },
    { path: '/batch', label: 'Toplu İşlem', icon: <Layers size={20} /> },
    { path: '/history', label: 'Geçmiş / Veritabanı', icon: <Database size={20} /> },
    { path: '/reporting', label: 'Raporlama', icon: <FileText size={20} /> },
    { path: '#', label: 'Ayarlar', icon: <Settings size={20} /> },
  ];

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="sidebar-header">
          Deep<span>Embryo</span>
        </div>
        <nav className="sidebar-nav">
          {menuItems.map((item, idx) => (
            <NavLink
              key={idx}
              to={item.path}
              className={`nav-item ${location.pathname === item.path ? 'active' : ''}`}
            >
              {item.icon}
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      
      <main className="main-content">
        <header className="topbar">
          <div className="topbar-title" style={{fontWeight: 600}}>
            Klinik Karar Destek Sistemi
          </div>
          <div className="topbar-user">
            <span>Dr. Embriyolog</span>
            <div className="topbar-avatar"><User size={18}/></div>
          </div>
        </header>
        <div className="page-container">
          {children}
        </div>
      </main>
    </div>
  );
}
