import React, { useState } from 'react';
import WelcomePage from './WelcomePage';
import RobotMotionControl from './RobotMotionControl';

function App() {
  const [showWelcome, setShowWelcome] = useState(true);

  const handleStart = () => {
    setShowWelcome(false);
  };

  return (
    <div className="App">
      {showWelcome ? (
        <WelcomePage onStart={handleStart} />
      ) : (
        <RobotMotionControl />
      )}
    </div>
  );
}

export default App;