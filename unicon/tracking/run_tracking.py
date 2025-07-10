#!/usr/bin/env python3
"""
Example usage of the multi-camera robot tracking system
"""

import time
import json
import threading

import numpy as np

from unicon.tracking.multi_camera_tracker import MultiCameraTracker, RobotClient, CameraConfig


def load_camera_configs(calibration_file: str = "camera_calibration.json") -> list:
    """Load camera configurations from calibration file"""

    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)
    
    configs = []
    
    # Reference camera (assumed at origin)
    if calib_data:
        first_serial = list(calib_data.keys())[0]
        if first_serial != 'relative_poses':
            configs.append(CameraConfig(
                serial=first_serial,
                position=np.array([0, 0, 1]),  # 1 meter high at origin
                rotation=np.eye(3)
            ))
    
    # Other cameras based on relative poses
    if 'relative_poses' in calib_data:
        poses = calib_data['relative_poses']['poses']
        for serial, pose in poses.items():
            rotation = np.array(pose['rotation'])
            translation = np.array(pose['translation'])
            
            configs.append(CameraConfig(
                serial=serial,
                position=translation,
                rotation=rotation
            ))
    
    return configs


def robot_control_loop(robot_id: str, tracker_address: str = "tcp://localhost:5555"):
    """Example robot control loop"""

    client = RobotClient(robot_id, tracker_address)
    
    try:
        while True:
            # Get current state
            state = client.get_state()
            
            if 'error' not in state:
                position = np.array(state['position'])
                velocity = np.array(state['velocity'])
                avoidance = np.array(state['avoidance_velocity'])
                
                # Simple control logic
                # In real implementation, this would control actual robot motors
                desired_velocity = velocity + 0.5 * avoidance  # Blend current and avoidance
                
                print(f"{robot_id}: pos={position}, vel={velocity}, avoid={avoidance}")
                
                # TODO(ming): Here you would send motor commands to the actual robot
                # For example, using serial communication or another protocol
                
            time.sleep(0.1)  # 10 Hz control loop
            
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


def main():
    """Main example"""
    
    # Load camera configurations
    try:
        camera_configs = load_camera_configs()
    except FileNotFoundError:
        print("No calibration file found. Using default configuration.")
        # Default configuration for testing
        camera_configs = [
            CameraConfig(
                serial="123456",  # Replace with your camera serial
                position=np.array([0, 0, 1]),
                rotation=np.eye(3)
            ),
            CameraConfig(
                serial="789012",  # Replace with your camera serial
                position=np.array([2, 0, 1]),
                rotation=np.eye(3)
            )
        ]
    
    # Create tracker
    tracker = MultiCameraTracker(camera_configs, zmq_port=5555)
    
    # Define robot colors (HSV ranges)
    # You'll need to calibrate these for your specific colored markers
    robot_colors = {
        "robot_red": (
            np.array([0, 100, 100]),    # Lower HSV
            np.array([10, 255, 255])    # Upper HSV
        ),
        "robot_blue": (
            np.array([100, 100, 100]),
            np.array([130, 255, 255])
        ),
        "robot_green": (
            np.array([40, 100, 100]),
            np.array([80, 255, 255])
        ),
        "robot_yellow": (
            np.array([20, 100, 100]),
            np.array([30, 255, 255])
        )
    }
    
    # Register robots
    for robot_id, color_range in robot_colors.items():
        tracker.register_robot(robot_id, color_range, radius=30)
    
    # Start tracker
    tracker.start()
    print("Tracker started. Press Ctrl+C to stop.")
    
    # Start robot control threads
    robot_threads = []
    for robot_id in robot_colors.keys():
        thread = threading.Thread(
            target=robot_control_loop,
            args=(robot_id,),
            daemon=True
        )
        thread.start()
        robot_threads.append(thread)
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
            
            # Optional: Print overall system status
            # You could add visualization here using OpenCV or matplotlib
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        tracker.stop()
        print("Tracker stopped.")


if __name__ == "__main__":
    main()