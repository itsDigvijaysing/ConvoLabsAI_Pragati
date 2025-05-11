// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import './App.css'; // Keep for any global styles or clean up

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/*" element={<Dashboard />} /> {/* Render Dashboard for all paths */}
      </Routes>
    </Router>
  );
}

export default App;