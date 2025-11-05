import React, { useState, useEffect } from 'react';
import { Plus, Trash2, Clock, AlertCircle, CheckCircle, Zap } from 'lucide-react';
import './App.css';

const API_BASE = 'http://localhost:8000';

const RobotMotionControl = () => {
  const [robotModels, setRobotModels] = useState([]);
  const [selectedRobot, setSelectedRobot] = useState('');
  const [motionType, setMotionType] = useState('Pick & Place');
  const [jointId, setJointId] = useState(1);
  const [currentPos, setCurrentPos] = useState(0.0);
  const [desiredPos, setDesiredPos] = useState(45.0);
  const [desiredVel, setDesiredVel] = useState(2.0);
  const [jointQueue, setJointQueue] = useState([]);
  const [timeResult, setTimeResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  useEffect(() => {
    fetchRobotModels();
  }, []);

  const fetchRobotModels = async () => {
    try {
      const response = await fetch(`${API_BASE}/robot-models`);
      const data = await response.json();
      setRobotModels(data.models);
      if (data.models.length > 0) setSelectedRobot(data.models[0]);
    } catch (err) {
      setError('Failed to load robot models. Make sure backend is running.');
    }
  };

  const addJoint = async () => {
    setError('');
    setSuccess('');
    
    if (jointQueue.length > 0) {
      if (jointQueue[0].robot_model !== selectedRobot) {
        setError('Cannot mix different robot models in one task!');
        return;
      }
      if (jointQueue[0].motion_type !== motionType) {
        setError('Cannot mix different motion types in one task!');
        return;
      }
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          robot_model: selectedRobot,
          motion_type: motionType,
          joint_id: jointId,
          current_position: currentPos,
          desired_position: desiredPos,
          desired_velocity: desiredVel
        })
      });

      const data = await response.json();
      
      if (response.ok) {
        setJointQueue([...jointQueue, data]);
        setSuccess(`Joint ${jointId} added successfully!`);
        setTimeResult(null);
      } else {
        setError(data.detail || 'Prediction failed');
      }
    } catch (err) {
      setError('Failed to add joint. Check backend connection.');
    } finally {
      setLoading(false);
    }
  };

  const clearJoints = () => {
    setJointQueue([]);
    setTimeResult(null);
    setSuccess('All joints cleared!');
    setError('');
  };

  const calculateTime = async () => {
    if (jointQueue.length === 0) {
      setError('No joint movements added!');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${API_BASE}/calculate-time`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ joints: jointQueue })
      });

      const data = await response.json();
      
      if (response.ok) {
        setTimeResult(data);
        setSuccess('Time calculation completed!');
      } else {
        setError(data.detail || 'Calculation failed');
      }
    } catch (err) {
      setError('Failed to calculate time. Check backend connection.');
    } finally {
      setLoading(false);
    }
  };

  const getPerformanceRating = (time) => {
    if (time < 2) return { label: 'VERY FAST', color: 'rating-very-fast', icon: <Zap className="icon-sm" /> };
    if (time < 4) return { label: 'FAST', color: 'rating-fast', icon: <CheckCircle className="icon-sm" /> };
    if (time < 6) return { label: 'MODERATE', color: 'rating-moderate', icon: <Clock className="icon-sm" /> };
    if (time < 10) return { label: 'SLOW', color: 'rating-slow', icon: <AlertCircle className="icon-sm" /> };
    return { label: 'VERY SLOW', color: 'rating-very-slow', icon: <AlertCircle className="icon-sm" /> };
  };

  return (
    <div className="app-container">
      <div className="content-wrapper">
        <h1 className="app-title">
          🤖 Robot Motion Control Predictor
        </h1>
        <p className="app-subtitle">AI-Powered Motion Planning & Time Estimation</p>

        <div className="grid-container">
          {/* Input Panel */}
          <div className="card control-panel">
            <h2 className="card-title">Control Panel</h2>
            
            <div className="info-box">
              <p className="info-text">
                <strong>ℹ️ Info:</strong> {motionType === 'Pick & Place' 
                  ? 'Pick & Place motions focus on accurate positioning.' 
                  : 'Weld & Inspect motions require precise velocity control.'}
              </p>
            </div>

            {error && (
              <div className="alert alert-error">
                <p>{error}</p>
              </div>
            )}

            {success && (
              <div className="alert alert-success">
                <p>{success}</p>
              </div>
            )}

            <div className="form-group-container">
              <div className="form-group">
                <label className="form-label">Robot Model</label>
                <select 
                  value={selectedRobot}
                  onChange={(e) => setSelectedRobot(e.target.value)}
                  className="form-select"
                >
                  {robotModels.map(model => (
                    <option key={model} value={model}>{model}</option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Motion Type</label>
                <select 
                  value={motionType}
                  onChange={(e) => setMotionType(e.target.value)}
                  className="form-select"
                >
                  <option value="Pick & Place">Pick & Place</option>
                  <option value="Weld & Inspect">Weld & Inspect</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Joint ID (1-6)</label>
                <input
                  type="number"
                  min="1"
                  max="6"
                  value={jointId}
                  onChange={(e) => setJointId(parseInt(e.target.value))}
                  className="form-input"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Current Position (°)</label>
                <input
                  type="number"
                  step="0.1"
                  value={currentPos}
                  onChange={(e) => setCurrentPos(parseFloat(e.target.value))}
                  className="form-input"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Desired Position (°)</label>
                <input
                  type="number"
                  step="0.1"
                  value={desiredPos}
                  onChange={(e) => setDesiredPos(parseFloat(e.target.value))}
                  className="form-input"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Desired Velocity (°/s)</label>
                <input
                  type="number"
                  step="0.1"
                  value={desiredVel}
                  onChange={(e) => setDesiredVel(parseFloat(e.target.value))}
                  className="form-input"
                />
              </div>

              <div className="button-row">
                <button
                  onClick={addJoint}
                  disabled={loading}
                  className="btn btn-primary"
                >
                  <Plus className="icon-sm" />
                  Add Joint
                </button>
                <button
                  onClick={clearJoints}
                  className="btn btn-warning"
                >
                  <Trash2 className="icon-sm" />
                </button>
              </div>

              <button
                onClick={calculateTime}
                disabled={loading || jointQueue.length === 0}
                className="btn btn-success btn-full"
              >
                <Clock className="icon-sm" />
                Calculate Total Time
              </button>
            </div>
          </div>

          {/* Results Panel */}
          <div className="results-panel">
            {/* Joint Queue */}
            <div className="card">
              <h2 className="card-title">Joint Queue</h2>
              
              {jointQueue.length === 0 ? (
                <div className="empty-state">
                  No joints added yet
                </div>
              ) : (
                <div className="joint-list">
                  {jointQueue.map((joint, idx) => (
                    <div key={idx} className="joint-item">
                      <div className="joint-header">Joint {joint.joint_id}</div>
                      <div className="joint-details">
                        <div className="detail-row">
                          <strong>Input:</strong> Desired Pos: {joint.desired_position.toFixed(2)}°, 
                          Desired Vel: {joint.desired_velocity.toFixed(2)}°/s
                        </div>
                        <div className="detail-row detail-predicted">
                          <strong>Predicted:</strong> Actual Pos: {joint.predicted_position.toFixed(2)}°, 
                          Actual Vel: {joint.predicted_velocity.toFixed(2)}°/s
                        </div>
                        <div className="detail-row">
                          <strong>Movement:</strong> {joint.current_position.toFixed(2)}° → {joint.predicted_position.toFixed(2)}°
                        </div>
                        <div className="detail-row detail-error">
                          <strong>Errors:</strong> Pos: {joint.position_error.toFixed(4)}° ({joint.position_error_pct.toFixed(2)}%), 
                          Vel: {joint.velocity_error.toFixed(4)}°/s ({joint.velocity_error_pct.toFixed(2)}%)
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Time Results */}
            {timeResult && (
              <div className="card">
                <h2 className="card-title">Time Estimation Report</h2>
                
                <div className="time-results">
                  <div className="task-info">
                    <div className="info-item">
                      <strong>Robot:</strong> {timeResult.robot_model}
                    </div>
                    <div className="info-item">
                      <strong>Motion:</strong> {timeResult.motion_type}
                    </div>
                    <div className="info-item">
                      <strong>Joints:</strong> {timeResult.num_joints}
                    </div>
                  </div>

                  <div className="total-time-box">
                    <div className="total-time-label">TOTAL ESTIMATED TIME</div>
                    <div className="total-time-value">
                      {timeResult.total_time.toFixed(3)}s
                    </div>
                    <div className="total-time-minutes">
                      ({(timeResult.total_time / 60).toFixed(2)} minutes)
                    </div>
                  </div>

                  <div className="metrics-grid">
                    <div className="metric-box">
                      <div className="metric-label">Movement Time</div>
                      <div className="metric-value">
                        {timeResult.breakdown.movement_time.toFixed(3)}s
                      </div>
                    </div>
                    <div className="metric-box">
                      <div className="metric-label">Overhead Time</div>
                      <div className="metric-value">
                        {timeResult.breakdown.overhead_time.toFixed(3)}s
                      </div>
                    </div>
                    <div className="metric-box">
                      <div className="metric-label">Max Joint Time</div>
                      <div className="metric-value">
                        {timeResult.breakdown.max_joint_time.toFixed(3)}s
                      </div>
                    </div>
                    <div className="metric-box metric-highlight">
                      <div className="metric-label">Efficiency Gain</div>
                      <div className="metric-value">
                        {timeResult.efficiency_gain.toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  <div className="performance-rating">
                    <span className="rating-label">Performance Rating:</span>
                    <div className={`rating-value ${getPerformanceRating(timeResult.total_time).color}`}>
                      {getPerformanceRating(timeResult.total_time).icon}
                      {getPerformanceRating(timeResult.total_time).label}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RobotMotionControl;