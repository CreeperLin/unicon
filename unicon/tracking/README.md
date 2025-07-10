# Multi-Camera Robot Tracking

This system tracks multiple robots using colored markers through multiple Intel RealSense cameras. It provides real-time position tracking, collision prediction, and avoidance suggestions via ZMQ communication.

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ RealSense Cam 1 │     │ RealSense Cam 2 │     │ RealSense Cam N │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    MultiCameraTracker   │
                    │  - Color Detection      │
                    │  - Position Tracking    │
                    │  - Collision Detection  │
                    └────────────┬────────────┘
                                 │
                            ZMQ (Port 5555)
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌────────▼────────┐     ┌────────▼────────┐     ┌────────▼────────┐
│   Robot 1       │     │   Robot 2       │     │   Robot N       │
│  (Red Marker)   │     │  (Blue Marker)  │     │ (Green Marker)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Prerequisites

### Hardware
- 2+ Intel RealSense cameras (D415, D435, D455, etc.)
- Colored markers for robots (distinct colors)
- Computer with USB 3.0 ports
- Robots with ZMQ client capability

### Software
```bash
# Install required packages
pip install opencv-python numpy pyrealsense2 pyzmq filterpy scipy

# Optional for visualization
pip install matplotlib
```

## Setup Process

### Step 1: Camera Calibration

First, calibrate your cameras to find their relative positions:

```bash
python camera_calibration.py
```

1. **Print a checkerboard pattern** (9x6 inner corners, 25mm squares)
2. **Capture calibration images** for each camera (15-20 images)
3. **Find relative poses** by placing the checkerboard visible to all cameras
4. The calibration will be saved to `camera_calibration.json`

### Step 2: Color Calibration

Calibrate the HSV ranges for your robot markers:

```bash
python color_calibration.py
```

1. Place your robot markers in view of the camera
2. Adjust HSV sliders to isolate each color
3. Click on the marker to auto-adjust ranges
4. Save calibration for each robot (press 'S')
5. Calibrations saved to `color_calibrations.json`

### Step 3: Configure the Tracker

Edit the tracker configuration in your main script:

```python
# Load calibrations
camera_configs = load_camera_configs("camera_calibration.json")
color_calibrations = load_color_calibrations("color_calibrations.json")

# Create tracker
tracker = MultiCameraTracker(camera_configs, zmq_port=5555)

# Register robots
for robot_name, color_data in color_calibrations.items():
    tracker.register_robot(
        robot_id=robot_name,
        color_hsv_range=(
            np.array(color_data['hsv_lower']),
            np.array(color_data['hsv_upper'])
        ),
        radius=30  # Robot radius in mm
    )
```

### Step 4: Start the System

```bash
python tracker_example.py
```

The tracker will:
- Start all cameras
- Begin tracking colored markers
- Provide position/velocity data via ZMQ
- Calculate collision avoidance suggestions

## Robot Client Implementation

Each robot should implement a ZMQ client:

```python
from multi_camera_tracker import RobotClient
import numpy as np

# Create client
client = RobotClient("robot_1", "tcp://tracker_ip:5555")

# Main control loop
while True:
    # Get current state
    state = client.get_state()
    
    position = np.array(state['position'])
    velocity = np.array(state['velocity'])
    avoidance = np.array(state['avoidance_velocity'])
    
    # Apply control logic
    desired_velocity = velocity + 0.5 * avoidance
    
    # Send commands to motors
    control_motors(desired_velocity)
    
    time.sleep(0.1)  # 10 Hz
```

## ZMQ Protocol

### Request: Get Robot State
```json
{
    "type": "get_state",
    "robot_id": "robot_1"
}
```

### Response: Robot State
```json
{
    "position": [x, y],
    "velocity": [vx, vy],
    "avoidance_velocity": [ax, ay],
    "timestamp": 1234567890.123
}
```

### Request: Get All Robots
```json
{
    "type": "get_all_robots"
}
```

### Response: All Robots
```json
{
    "robot_1": {
        "position": [x1, y1],
        "velocity": [vx1, vy1]
    },
    "robot_2": {
        "position": [x2, y2],
        "velocity": [vx2, vy2]
    }
}
```

## Troubleshooting

### Camera Not Found
- Check USB 3.0 connection
- Verify camera serial numbers in configuration
- Use `rs-enumerate-devices` to list cameras

### Poor Tracking Performance
- Ensure good lighting conditions
- Check color calibration ranges
- Verify marker size is sufficient
- Reduce camera exposure for better color detection

### Position Jumps
- Calibrate cameras more accurately
- Increase Kalman filter smoothing
- Check for reflections or similar colors

### Collision Detection Issues
- Adjust `collision_threshold` parameter
- Verify robot radius settings
- Check prediction horizon settings

## Performance Optimization

1. **Camera Settings**
   - Lower resolution for faster processing (640x480)
   - Adjust exposure for marker visibility
   - Disable auto-exposure

2. **Tracking Parameters**
   ```python
   tracker.min_contour_area = 100  # Minimum marker size
   tracker.max_tracking_distance = 100  # Max movement between frames
   tracker.collision_threshold = 50  # Safety margin
   ```

3. **Network Optimization**
   - Use wired connection for tracker computer
   - Minimize ZMQ message size
   - Consider UDP for lower latency

## Advanced Features

### Custom Collision Avoidance
Override the default avoidance algorithm:

```python
def custom_avoidance(robot, obstacles):
    # Implement your algorithm
    return avoidance_velocity

tracker._calculate_avoidance_velocity = custom_avoidance
```

### Multiple Marker Tracking
Track robots with multiple markers:

```python
# Register same robot with multiple colors
tracker.register_robot("robot_1", red_hsv_range)
tracker.register_robot("robot_1_marker2", green_hsv_range)

# Combine in post-processing
```

### 3D Tracking
Utilize depth information:

```python
def _pixel_to_world_3d(self, pixel, camera_config, depth):
    # Return full 3D coordinates
    return np.array([x, y, z])
```

## Safety Considerations

1. **Emergency Stop**: Implement client-side emergency stop
2. **Timeout Handling**: Detect tracker communication loss
3. **Boundary Checking**: Define safe operating area
4. **Manual Override**: Allow manual control override

## Example Applications

1. **Swarm Robotics**: Coordinate multiple robots
2. **Warehouse Automation**: Track inventory robots
3. **Research**: Multi-agent path planning
4. **Education**: Demonstrate collision avoidance

## Support

For issues or questions:
1. Check camera connections and calibration
2. Verify color ranges are distinct
3. Monitor ZMQ communication
4. Enable debug logging in tracker