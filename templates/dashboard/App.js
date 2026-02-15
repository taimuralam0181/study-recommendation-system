/**
 * Cyberpunk Dashboard - React Native for Web
 * This file initializes the React app and connects it to the Django API
 */

(function() {
  'use strict';
  
  // API Configuration
  const API_BASE = '/dashboard/api';
  
  // Mock data for fallback
  const MOCK_DATA = {
    student: {
      username: 'john_doe',
      cgpa: 3.75,
      level: 'Advanced',
      current_semester: 4,
      total_semesters: 8,
    },
    ml_prediction: {
      predicted_grade: 'A-',
      confidence: 87.5,
      probabilities: { 'A+': 12, A: 25, 'A-': 40, 'B+': 15, B: 5, F: 8 },
      risk_level: 'LOW',
      recovery_possible: true,
    },
    semester_performance: [
      { subject: 'DSA', midterm: 85, prefinal: 82, predicted: 'A', target: 'A', status: 'ON_TRACK' },
      { subject: 'ALGO', midterm: 72, prefinal: 78, predicted: 'B+', target: 'A-', status: 'ATTENTION' },
      { subject: 'DBMS', midterm: 65, prefinal: 71, predicted: 'B', target: 'B+', status: 'ATTENTION' },
      { subject: 'OS', midterm: 45, prefinal: 58, predicted: 'C+', target: 'B-', status: 'AT_RISK' },
      { subject: 'MATH', midterm: 90, prefinal: 88, predicted: 'A+', target: 'A+', status: 'ON_TRACK' },
    ],
    study_recommendations: {
      materials: [
        { id: 1, title: 'Operating Systems - Process Scheduling', priority: 'HIGH' },
        { id: 2, title: 'Database Systems - Normalization', priority: 'HIGH' },
        { id: 3, title: 'Algorithms - Dynamic Programming', priority: 'MEDIUM' },
      ],
      focus_areas: [
        { subject: 'OS', action: 'Score 15 more points needed', type: 'RECOVERY' },
        { subject: 'DBMS', action: 'Practice more queries', type: 'IMPROVEMENT' },
      ],
    },
    recent_activity: [
      { date: 'TODAY', action: 'Entered marks for OS', details: 'Midterm: 45/30' },
      { date: 'TODAY', action: 'ML Prediction updated', details: 'Risk level: LOW' },
      { date: 'YESTERDAY', action: 'Viewed recovery plan', details: 'DBMS' },
    ],
  };
  
  // Theme Configuration
  const THEME = {
    colors: {
      background: '#0A0A0F',
      backgroundSecondary: '#12121A',
      neonCyan: '#00F0FF',
      neonPink: '#FF0080',
      neonPurple: '#B829E0',
      neonGreen: '#00FF88',
      textPrimary: '#FFFFFF',
      textSecondary: '#8892A0',
    },
  };
  
  // API Functions
  async function fetchDashboardData() {
    try {
      const response = await fetch(API_BASE + '/');
      if (!response.ok) {
        throw new Error('API request failed');
      }
      return await response.json();
    } catch (error) {
      console.warn('Using mock data:', error.message);
      return MOCK_DATA;
    }
  }
  
  async function predictGrade(data) {
    try {
      const response = await fetch(API_BASE + '/predict/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      if (!response.ok) {
        throw new Error('Prediction failed');
      }
      return await response.json();
    } catch (error) {
      console.error('Prediction error:', error);
      throw error;
    }
  }
  
  // React Components (Using vanilla JS for web compatibility)
  function createElement(type, props, children) {
    const element = document.createElement(type);
    
    if (props) {
      Object.keys(props).forEach(key => {
        if (key === 'style' && typeof props[key] === 'object') {
          Object.assign(element.style, props[key]);
        } else if (key === 'className') {
          element.className = props[key];
        } else if (key === 'onClick') {
          element.addEventListener('click', props[key]);
        } else if (key.startsWith('on')) {
          element.addEventListener(key.slice(2).toLowerCase(), props[key]);
        } else {
          element.setAttribute(key, props[key]);
        }
      });
    }
    
    if (children) {
      children.forEach(child => {
        if (typeof child === 'string') {
          element.appendChild(document.createTextNode(child));
        } else if (child instanceof Node) {
          element.appendChild(child);
        }
      });
    }
    
    return element;
  }
  
  // Dashboard Component
  async function Dashboard() {
    const data = await fetchDashboardData();
    const { student, ml_prediction, semester_performance, study_recommendations, recent_activity } = data;
    
    const container = createElement('div', { className: 'dashboard-container' });
    
    // Stats Row
    const statsRow = createElement('div', { className: 'stats-grid' });
    statsRow.appendChild(createStatCard('CGPA', student.cgpa.toFixed(2), '↑ 0.2 this sem', 'cyan'));
    statsRow.appendChild(createStatCard('LEVEL', student.level, 'Performance: A+', 'purple'));
    statsRow.appendChild(createStatCard('SEMESTER', `${student.current_semester}/${student.total_semesters}`, 'Current', 'green'));
    statsRow.appendChild(createStatCard('STATUS', 'ACTIVE', 'All systems OK', 'green'));
    
    container.appendChild(statsRow);
    
    // ML Prediction Panel
    container.appendChild(createMLPanel(ml_prediction));
    
    // Performance Table
    container.appendChild(createPerformanceTable(semester_performance));
    
    // Recommendations & Activity
    const bottomGrid = createElement('div', { className: 'grid-2' });
    bottomGrid.appendChild(createRecommendationsPanel(study_recommendations));
    bottomGrid.appendChild(createActivityPanel(recent_activity));
    container.appendChild(bottomGrid);
    
    // Quick Actions
    container.appendChild(createQuickActions());
    
    return container;
  }
  
  function createStatCard(label, value, subtext, color) {
    const card = createElement('div', { className: `stat-card ${color}` });
    card.innerHTML = `
      <div class="stat-label">${label}</div>
      <div class="stat-value ${color}">${value}</div>
      <div class="stat-subtext">${subtext}</div>
    `;
    return card;
  }
  
  function createMLPanel(prediction) {
    const panel = createElement('div', { className: 'cyber-card pink' });
    
    let probabilitiesHTML = '';
    Object.entries(prediction.probabilities).forEach(([grade, pct]) => {
      const barColor = grade === 'F' ? 'pink' : 'cyan';
      probabilitiesHTML += `
        <div class="progress-container">
          <div class="progress-header">
            <span class="progress-label">${grade}</span>
            <span class="progress-value">${pct}%</span>
          </div>
          <div class="progress-bar">
            <div class="progress-fill ${barColor}" style="width: ${pct}%;"></div>
          </div>
        </div>
      `;
    });
    
    panel.innerHTML = `
      <h2 class="card-title pink">⚡ ML Prediction Engine</h2>
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
        <span style="color: #8892A0; font-size: 0.85rem;">
          Predicted Grade: <strong style="color: #FF0080; font-size: 1.5rem;">${prediction.predicted_grade}</strong>
        </span>
        <span style="color: #8892A0; font-size: 0.85rem;">
          Confidence: <strong style="color: #00F0FF;">${prediction.confidence}%</strong>
        </span>
      </div>
      ${probabilitiesHTML}
      <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #1A3A4A;">
        <span style="color: #8892A0;">Risk Level: </span>
        <strong style="color: #00FF88;">${prediction.risk_level}</strong>
        <span style="color: #8892A0; margin-left: 15px;">Recovery: </span>
        <strong style="color: #00FF88;">${prediction.recovery_possible ? 'POSSIBLE' : 'NOT AVAILABLE'}</strong>
      </div>
    `;
    
    return panel;
  }
  
  function createPerformanceTable(performances) {
    const panel = createElement('div', { className: 'cyber-card cyan' });
    
    let rowsHTML = performances.map(p => {
      const statusClass = {
        'ON_TRACK': 'on-track',
        'ATTENTION': 'attention',
        'AT_RISK': 'at-risk',
      }[p.status] || '';
      
      const statusIcon = {
        'ON_TRACK': '✓',
        'ATTENTION': '!',
        'AT_RISK': '⚠',
      }[p.status] || '';
      
      return `
        <tr>
          <td>${p.subject}</td>
          <td>${p.midterm}</td>
          <td>${p.prefinal}</td>
          <td>${p.predicted}</td>
          <td style="color: #00F0FF;">${p.target}</td>
          <td><span class="status-badge ${statusClass}">${statusIcon}</span></td>
        </tr>
      `;
    }).join('');
    
    panel.innerHTML = `
      <h2 class="card-title cyan">📊 Semester Performance Matrix</h2>
      <table class="performance-table">
        <thead>
          <tr>
            <th>SUBJECT</th>
            <th>MIDTERM</th>
            <th>PREFINAL</th>
            <th>PREDICTED</th>
            <th>TARGET</th>
            <th>STATUS</th>
          </tr>
        </thead>
        <tbody>${rowsHTML}</tbody>
      </table>
    `;
    
    return panel;
  }
  
  function createRecommendationsPanel(recommendations) {
    const panel = createElement('div', { className: 'cyber-card pink' });
    
    const materialsHTML = recommendations.materials.map(m => `
      <li class="material-item">
        <div class="material-info">
          <div class="priority-dot ${m.priority.toLowerCase()}"></div>
          <span class="material-title">${m.title}</span>
        </div>
        <span class="priority-badge ${m.priority.toLowerCase()}">${m.priority}</span>
      </li>
    `).join('');
    
    const focusHTML = recommendations.focus_areas.map(f => `
      <p style="color: #8892A0; font-size: 0.8rem; margin-bottom: 8px;">
        • <strong style="color: #FF0080;">${f.subject}:</strong> ${f.action}
      </p>
    `).join('');
    
    panel.innerHTML = `
      <h2 class="card-title pink">📚 Study Recommendations</h2>
      <ul class="material-list">${materialsHTML}</ul>
      <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #1A3A4A;">
        <h4 style="color: #FF0080; font-size: 0.85rem; margin-bottom: 10px;">🎯 Focus Areas</h4>
        ${focusHTML}
      </div>
    `;
    
    return panel;
  }
  
  function createActivityPanel(activities) {
    const panel = createElement('div', { className: 'cyber-card green' });
    
    const activitiesHTML = activities.map(a => `
      <div class="activity-item">
        <span class="activity-date">[${a.date}]</span>
        <span class="activity-action">${a.action}</span>
        <span class="activity-details"> - ${a.details}</span>
      </div>
    `).join('');
    
    panel.innerHTML = `
      <h2 class="card-title green">📋 Activity Log</h2>
      ${activitiesHTML}
    `;
    
    return panel;
  }
  
  function createQuickActions() {
    const container = createElement('div', { className: 'quick-actions' });
    
    const buttons = [
      { text: 'Start Recovery Plan', color: 'pink' },
      { text: 'View Materials', color: 'cyan' },
      { text: 'Get Help', color: 'purple' },
    ];
    
    buttons.forEach(btn => {
      const button = createElement('button', {
        className: `cyber-button ${btn.color}`,
      }, [btn.text]);
      button.addEventListener('click', () => {
        alert(`${btn.text} clicked!`);
      });
      container.appendChild(button);
    });
    
    return container;
  }
  
  // Initialize App
  async function init() {
    const appRoot = document.getElementById('react-app');
    if (appRoot) {
      const dashboard = await Dashboard();
      appRoot.appendChild(dashboard);
    }
  }
  
  // Run when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  
  // Expose API functions globally
  window.dashboardAPI = {
    fetchDashboardData,
    predictGrade,
  };
  
})();
