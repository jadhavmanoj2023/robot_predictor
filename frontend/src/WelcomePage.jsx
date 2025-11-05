import React from 'react';
import './WelcomePage.css';

const WelcomePage = ({ onStart }) => {
  return (
    <div className="welcome-fullscreen">
      <div className="welcome-content-center">
        <h1 className="welcome-heading">Welcome</h1>
        <p className="subtitle">Robot Assistant AI</p>
        <button className="start-btn" onClick={onStart}>
          Start
        </button>
      </div>
    </div>
  );
};

export default WelcomePage;
