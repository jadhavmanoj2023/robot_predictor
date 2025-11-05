"""
FastAPI Backend for Robot Motion Control System
Install dependencies: pip install fastapi uvicorn pandas scikit-learn numpy matplotlib pillow
Run: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
from pathlib import Path
import os

app = FastAPI(title="Robot Motion Control API")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and encoders
rf_position = None
rf_velocity = None
robot_encoder = None
data = None

# DH Parameters for Forward Kinematics
DH_PARAMS = {
    'link_lengths': [0.0, 0.425, 0.392, 0.0, 0.0, 0.0],
    'link_offsets': [0.163, 0.0, 0.0, 0.134, 0.100, 0.100]
}

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

class CoordinateData(BaseModel):
    desired: Dict[str, float]
    actual: Dict[str, float]
    error: Dict[str, float]
    euclidean_distance: float
    accuracy_rating: str

class VisualizationResponse(BaseModel):
    coordinates: CoordinateData
    plot_3d: str
    plot_xy: str
    plot_error: str

# Forward Kinematics Functions
def forward_kinematics(joint_angles):
    """Calculate end-effector position from joint angles (degrees)"""
    theta = [np.radians(angle) for angle in joint_angles]
    a = DH_PARAMS['link_lengths']
    d = DH_PARAMS['link_offsets']

    c1, s1 = np.cos(theta[0]), np.sin(theta[0])
    c2, s2 = np.cos(theta[1]), np.sin(theta[1])
    c23 = np.cos(theta[1] + theta[2])
    s23 = np.sin(theta[1] + theta[2])

    x = c1 * (a[1]*c2 + a[2]*c23 + d[3]*s23 + d[5]*np.sin(theta[1]+theta[2]+theta[3]))
    y = s1 * (a[1]*c2 + a[2]*c23 + d[3]*s23 + d[5]*np.sin(theta[1]+theta[2]+theta[3]))
    z = d[0] + a[1]*s2 + a[2]*s23 - d[3]*c23 - d[5]*np.cos(theta[1]+theta[2]+theta[3])

    return np.array([x, y, z])

def calculate_end_effector_coords(joint_positions):
    """Calculate end-effector from joint position dict"""
    angles = [joint_positions.get(i, 0.0) for i in range(1, 7)]
    return forward_kinematics(angles)

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
            "/calculate-time": "POST - Calculate task time",
            "/visualize": "POST - Generate visualization graphs"
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

@app.post("/visualize")
async def visualize_coordinates(request: TimeCalculationRequest):
    """Generate visualization graphs for end-effector coordinates"""
    if len(request.joints) == 0:
        raise HTTPException(status_code=400, detail="No joint movements provided")
    
    try:
        # Calculate desired and actual joint positions
        desired_joint_positions = {joint.joint_id: joint.desired_position for joint in request.joints}
        actual_joint_positions = {joint.joint_id: joint.predicted_position for joint in request.joints}
        
        # Calculate end-effector coordinates
        desired_coords = calculate_end_effector_coords(desired_joint_positions)
        actual_coords = calculate_end_effector_coords(actual_joint_positions)
        position_error = actual_coords - desired_coords
        euclidean_distance = np.linalg.norm(position_error)
        
        # Determine accuracy rating
        if euclidean_distance < 0.001:
            rating = "🟢 EXCELLENT (< 1mm)"
        elif euclidean_distance < 0.005:
            rating = "🟡 GOOD (< 5mm)"
        elif euclidean_distance < 0.010:
            rating = "🟠 ACCEPTABLE (< 10mm)"
        else:
            rating = "🔴 POOR (> 10mm)"
        
        # Create coordinate data
        coord_data = CoordinateData(
            desired={
                "x": float(desired_coords[0]),
                "y": float(desired_coords[1]),
                "z": float(desired_coords[2])
            },
            actual={
                "x": float(actual_coords[0]),
                "y": float(actual_coords[1]),
                "z": float(actual_coords[2])
            },
            error={
                "x": float(position_error[0] * 1000),  # Convert to mm
                "y": float(position_error[1] * 1000),
                "z": float(position_error[2] * 1000)
            },
            euclidean_distance=float(euclidean_distance * 1000),  # Convert to mm
            accuracy_rating=rating
        )
        

        # Generate plots correctly
        # --- 3D plot ---
        fig3d = plt.figure(figsize=(6, 5))
        ax1 = fig3d.add_subplot(111, projection='3d')
        ax1.scatter(desired_coords[0], desired_coords[1], desired_coords[2],
                c='blue', s=200, marker='o', label='Desired', alpha=0.7)
        ax1.scatter(actual_coords[0], actual_coords[1], actual_coords[2],
                c='red', s=200, marker='^', label='Actual', alpha=0.7)
        ax1.plot([desired_coords[0], actual_coords[0]],
                [desired_coords[1], actual_coords[1]],
                [desired_coords[2], actual_coords[2]],
                'g--', linewidth=2, label=f'Error: {euclidean_distance*1000:.2f}mm')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D End-Effector Position')
        ax1.legend()
        buf_3d = io.BytesIO()
        fig3d.savefig(buf_3d, format='png', bbox_inches='tight', dpi=100)
        plot_3d_base64 = base64.b64encode(buf_3d.getvalue()).decode('utf-8')
        plt.close(fig3d)

        # --- XY Plane Projection ---
        fig_xy = plt.figure(figsize=(6, 5))
        ax2 = fig_xy.add_subplot(111)
        ax2.scatter(desired_coords[0], desired_coords[1], c='blue', s=300, marker='o', label='Desired', alpha=0.7)
        ax2.scatter(actual_coords[0], actual_coords[1], c='red', s=300, marker='^', label='Actual', alpha=0.7)
        ax2.plot([desired_coords[0], actual_coords[0]], [desired_coords[1], actual_coords[1]], 'g--', linewidth=2)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY Plane Projection')
        ax2.legend()
        ax2.grid(True)
        buf_xy = io.BytesIO()
        fig_xy.savefig(buf_xy, format='png', bbox_inches='tight', dpi=100)
        plot_xy_base64 = base64.b64encode(buf_xy.getvalue()).decode('utf-8')
        plt.close(fig_xy)

        # --- Error Bars ---
        fig_err = plt.figure(figsize=(6, 5))
        ax3 = fig_err.add_subplot(111)
        axes_labels = ['X', 'Y', 'Z']
        errors_mm = position_error * 1000
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        ax3.bar(axes_labels, np.abs(errors_mm), color=colors, edgecolor='black')
        ax3.set_ylabel('Absolute Error (mm)')
        ax3.set_title('Position Error by Axis')
        ax3.grid(axis='y', alpha=0.3)
        for i, val in enumerate(errors_mm):
            ax3.text(i, abs(val) + 0.2, f"{val:+.2f}mm", ha='center')
        buf_err = io.BytesIO()
        fig_err.savefig(buf_err, format='png', bbox_inches='tight', dpi=100)
        plot_error_base64 = base64.b64encode(buf_err.getvalue()).decode('utf-8')
        plt.close(fig_err)

        # Return response
        return VisualizationResponse(
            coordinates=coord_data,
            plot_3d=f"data:image/png;base64,{plot_3d_base64}",
            plot_xy=f"data:image/png;base64,{plot_xy_base64}",
            plot_error=f"data:image/png;base64,{plot_error_base64}"
        )

        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Visualization error: {str(e)}")

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