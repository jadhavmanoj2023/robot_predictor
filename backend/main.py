"""
FastAPI Backend for Robot Motion Control System
Install dependencies: pip install fastapi uvicorn pandas scikit-learn numpy
Run: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = FastAPI(title="Robot Motion Control API")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and encoders
rf_position = None
rf_velocity = None
robot_encoder = None
data = None

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    robot_model: str
    motion_type: str
    joint_id: int
    current_position: float
    desired_position: float
    desired_velocity: float

class JointMovement(BaseModel):
    robot_model: str
    motion_type: str
    joint_id: int
    current_position: float
    desired_position: float
    desired_velocity: float
    predicted_position: float
    predicted_velocity: float
    position_error: float
    position_error_pct: float
    velocity_error: float
    velocity_error_pct: float

class TimeCalculationRequest(BaseModel):
    joints: List[JointMovement]

class TimeBreakdown(BaseModel):
    movement_time: float
    overhead_time: float
    max_joint_time: float
    sequential_time: float
    overlap_factor: float

class TimeCalculationResponse(BaseModel):
    total_time: float
    breakdown: TimeBreakdown
    robot_model: str
    motion_type: str
    num_joints: int
    efficiency_gain: float

# Initialize models on startup
@app.on_event("startup")
async def load_models():
    global rf_position, rf_velocity, robot_encoder, data
    
    print("🚀 Loading dataset and training models...")
    
    # Load dataset
    url = "https://raw.githubusercontent.com/cr1825/ai-project-data-sheet/main/EdgeAI_Robot_Motion_Control_Dataset_expanded_5k.csv"
    data = pd.read_csv(url)
    
    # Encode robot models
    robot_encoder = LabelEncoder()
    data['Robot_Model_Encoded'] = robot_encoder.fit_transform(data['Robot_Model'])
    
    # Separate data by motion type
    pick_place_data = data[data['Motion_Type'].isin(['Pick', 'Place'])].copy()
    weld_inspect_data = data[data['Motion_Type'].isin(['Weld', 'Inspect'])].copy()
    
    # Train Position Model (Pick & Place)
    X_position = pick_place_data[['Desired_Position', 'Desired_Velocity', 'Joint_ID', 'Robot_Model_Encoded']]
    y_position = pick_place_data['Actual_Position']
    
    X_pos_train, X_pos_test, y_pos_train, y_pos_test = train_test_split(
        X_position, y_position, test_size=0.2, random_state=42
    )
    
    rf_position = RandomForestRegressor(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1)
    rf_position.fit(X_pos_train, y_pos_train)
    
    # Train Velocity Model (Weld & Inspect)
    X_velocity = weld_inspect_data[['Desired_Position', 'Desired_Velocity', 'Joint_ID', 'Robot_Model_Encoded']]
    y_velocity = weld_inspect_data['Actual_Velocity']
    
    X_vel_train, X_vel_test, y_vel_train, y_vel_test = train_test_split(
        X_velocity, y_velocity, test_size=0.2, random_state=42
    )
    
    rf_velocity = RandomForestRegressor(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1)
    rf_velocity.fit(X_vel_train, y_vel_train)
    
    print("✅ Models loaded successfully!")
    print(f"   Available Robot Models: {list(robot_encoder.classes_)}")

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Robot Motion Control API",
        "version": "1.0",
        "endpoints": {
            "/robot-models": "GET - List available robot models",
            "/predict": "POST - Predict robot motion",
            "/calculate-time": "POST - Calculate task time"
        }
    }

@app.get("/robot-models")
async def get_robot_models():
    """Get list of available robot models"""
    if robot_encoder is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    return {
        "models": list(robot_encoder.classes_),
        "count": len(robot_encoder.classes_)
    }

@app.post("/predict")
async def predict_motion(request: PredictionRequest):
    """Predict actual position and velocity for a joint movement"""
    if rf_position is None or rf_velocity is None or robot_encoder is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Encode robot model
        robot_model_encoded = robot_encoder.transform([request.robot_model])[0]
        
        # Prepare input data
        input_data = pd.DataFrame({
            'Desired_Position': [request.desired_position],
            'Desired_Velocity': [request.desired_velocity],
            'Joint_ID': [request.joint_id],
            'Robot_Model_Encoded': [robot_model_encoded]
        })
        
        # Make predictions based on motion type
        if request.motion_type == 'Pick & Place':
            predicted_pos = float(rf_position.predict(input_data)[0])
            predicted_vel = request.desired_velocity
        else:  # Weld & Inspect
            predicted_vel = float(rf_velocity.predict(input_data)[0])
            predicted_pos = request.desired_position
        
        # Calculate errors
        position_error = abs(predicted_pos - request.desired_position)
        velocity_error = abs(predicted_vel - request.desired_velocity)
        
        position_error_pct = (position_error / abs(request.desired_position) * 100) if request.desired_position != 0 else 0
        velocity_error_pct = (velocity_error / abs(request.desired_velocity) * 100) if request.desired_velocity != 0 else 0
        
        return {
            "robot_model": request.robot_model,
            "motion_type": request.motion_type,
            "joint_id": request.joint_id,
            "current_position": request.current_position,
            "desired_position": request.desired_position,
            "desired_velocity": request.desired_velocity,
            "predicted_position": predicted_pos,
            "predicted_velocity": abs(predicted_vel) if predicted_vel != 0 else 1.0,
            "position_error": position_error,
            "position_error_pct": position_error_pct,
            "velocity_error": velocity_error,
            "velocity_error_pct": velocity_error_pct
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

def calculate_movement_time(current_pos: float, target_pos: float, velocity: float, 
                           acceleration: float = 10.0, deceleration: float = 10.0):
    """Calculate time for a joint to move from current to target position"""
    distance = abs(target_pos - current_pos)
    
    t_accel = velocity / acceleration
    t_decel = velocity / deceleration
    
    d_accel = 0.5 * acceleration * t_accel**2
    d_decel = 0.5 * deceleration * t_decel**2
    
    if (d_accel + d_decel) >= distance:
        v_peak = np.sqrt(2 * distance * acceleration * deceleration / (acceleration + deceleration))
        t_accel = v_peak / acceleration
        t_decel = v_peak / deceleration
        t_constant = 0
    else:
        d_constant = distance - d_accel - d_decel
        t_constant = d_constant / velocity
    
    total_time = t_accel + t_constant + t_decel
    
    return total_time, {
        'acceleration': t_accel,
        'constant_velocity': t_constant,
        'deceleration': t_decel
    }

@app.post("/calculate-time", response_model=TimeCalculationResponse)
async def calculate_task_time(request: TimeCalculationRequest):
    """Calculate total task time for all joint movements"""
    if len(request.joints) == 0:
        raise HTTPException(status_code=400, detail="No joint movements provided")
    
    try:
        movement_times = []
        
        for joint in request.joints:
            time, phases = calculate_movement_time(
                joint.current_position,
                joint.predicted_position,
                joint.predicted_velocity
            )
            movement_times.append(time)
        
        max_time = max(movement_times)
        total_sequential = sum(movement_times)
        overlap_factor = 0.7
        
        actual_movement_time = max_time + (1 - overlap_factor) * (total_sequential - max_time)
        
        # Calculate overhead based on motion type
        if request.joints[0].motion_type == 'Pick & Place':
            gripper_time = 0.5
            settling_time = 0.2
            overhead_time = gripper_time + settling_time
        else:  # Weld & Inspect
            setup_time = 0.3
            process_time = 1.0
            overhead_time = setup_time + process_time
        
        total_time = actual_movement_time + overhead_time
        
        # Calculate efficiency gain
        efficiency_gain = (1 - (max_time / total_sequential)) * 100 if total_sequential > 0 else 0
        
        return TimeCalculationResponse(
            total_time=total_time,
            breakdown=TimeBreakdown(
                movement_time=actual_movement_time,
                overhead_time=overhead_time,
                max_joint_time=max_time,
                sequential_time=total_sequential,
                overlap_factor=overlap_factor
            ),
            robot_model=request.joints[0].robot_model,
            motion_type=request.joints[0].motion_type,
            num_joints=len(request.joints),
            efficiency_gain=efficiency_gain
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Time calculation error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": rf_position is not None and rf_velocity is not None,
        "data_loaded": data is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)