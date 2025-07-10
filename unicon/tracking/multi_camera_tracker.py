from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

import time
import logging
import threading

import numpy as np
import pyrealsense2 as rs
import cv2
import zmq

from filterpy.kalman import KalmanFilter


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Robot:
    """Robot data structure"""

    id: str
    color_hsv_range: Tuple[np.ndarray, np.ndarray]  # (lower, upper) HSV bounds
    position: np.ndarray  # [x, y] in world coordinates
    velocity: np.ndarray  # [vx, vy]
    radius: float  # Robot radius for collision detection
    last_update: float
    kalman_filter: Optional[KalmanFilter] = None


@dataclass
class CameraConfig:
    """Camera configuration"""

    serial: str
    position: np.ndarray  # Camera position in world coordinates
    rotation: np.ndarray  # Camera rotation matrix
    intrinsics: Optional[rs.intrinsics] = None


class MultiCameraTracker:
    def __init__(self, camera_configs: List[CameraConfig], zmq_port: int = 5555):
        """
        Initialize multi-camera tracker
        
        Args:
            camera_configs: List of camera configurations
            zmq_port: Port for ZMQ communication
        """

        self.camera_configs = camera_configs
        self.pipelines = {}
        self.robots = {}
        self.zmq_port = zmq_port
        
        # Initialize cameras
        self._init_cameras()
        
        # Initialize ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{zmq_port}")
        
        # Tracking parameters
        self.min_contour_area = 100
        self.max_tracking_distance = 100  # pixels
        self.collision_threshold = 50  # minimum distance between robots
        self.prediction_horizon = 1.0  # seconds
        
        # Thread control
        self.running = False
        self.tracking_thread = None
        self.zmq_thread = None
        
    def _init_cameras(self):
        """Initialize Intel RealSense cameras"""
        ctx = rs.context()
        devices = ctx.query_devices()

        for config in self.camera_configs:
            pipeline = rs.pipeline()
            rs_config = rs.config()
            
            # Enable device by serial number
            rs_config.enable_device(config.serial)
            rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            try:
                profile = pipeline.start(rs_config)
                
                # Get camera intrinsics
                color_stream = profile.get_stream(rs.stream.color)
                intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
                config.intrinsics = intrinsics
                
                self.pipelines[config.serial] = pipeline
                logger.info(f"Initialized camera {config.serial}")
            except Exception as e:
                logger.error(f"Failed to initialize camera {config.serial}: {e}")
    
    def register_robot(self, robot_id: str, color_hsv_range: Tuple[np.ndarray, np.ndarray], 
                      radius: float = 20):
        """Register a new robot with its color range"""

        kf = self._create_kalman_filter()
        self.robots[robot_id] = Robot(
            id=robot_id,
            color_hsv_range=color_hsv_range,
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            radius=radius,
            last_update=time.time(),
            kalman_filter=kf
        )
        logger.info(f"Registered robot {robot_id}")
    
    def _create_kalman_filter(self) -> KalmanFilter:
        """Create Kalman filter for robot tracking"""

        kf = KalmanFilter(dim_x=4, dim_z=2)

        # State: [x, y, vx, vy]
        kf.x = np.array([0., 0., 0., 0.])

        # State transition matrix
        dt = 0.033  # ~30 FPS
        kf.F = np.array([[1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Measurement matrix
        kf.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])

        # Covariance matrices
        kf.R *= 10  # Measurement noise
        kf.Q *= 0.1  # Process noise
        kf.P *= 100  # Initial uncertainty

        return kf
    
    def _detect_colored_points(self, image: np.ndarray, hsv_range: Tuple[np.ndarray, np.ndarray]) -> List[np.ndarray]:
        """Detect colored points in image"""

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_range[0], hsv_range[1])
        
        # Morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points.append(np.array([cx, cy]))

        return points
    
    def _pixel_to_world(self, pixel: np.ndarray, camera_config: CameraConfig, depth: float) -> np.ndarray:
        """Convert pixel coordinates to world coordinates"""

        if camera_config.intrinsics is None:
            return pixel
        
        # Deproject pixel to 3D point in camera coordinates
        intrinsics = camera_config.intrinsics
        x = (pixel[0] - intrinsics.ppx) / intrinsics.fx * depth
        y = (pixel[1] - intrinsics.ppy) / intrinsics.fy * depth
        z = depth
        
        point_camera = np.array([x, y, z])
        
        # Transform to world coordinates
        point_world = camera_config.rotation @ point_camera + camera_config.position
        
        # Return 2D world coordinates (assuming z is up)
        return point_world[:2]
    
    def _update_robot_position(self, robot: Robot, measurements: List[np.ndarray]):
        """Update robot position using Kalman filter"""

        if len(measurements) == 0:
            # Predict only
            robot.kalman_filter.predict()
        else:
            # Average measurements if multiple
            measurement = np.mean(measurements, axis=0)
            
            # Update Kalman filter
            robot.kalman_filter.predict()
            robot.kalman_filter.update(measurement)
        
        # Extract state
        state = robot.kalman_filter.x
        robot.position = state[:2]
        robot.velocity = state[2:]
        robot.last_update = time.time()
    
    def _predict_collision(self, robot1: Robot, robot2: Robot) -> Optional[float]:
        """Predict collision time between two robots"""

        # Relative position and velocity
        rel_pos = robot2.position - robot1.position
        rel_vel = robot2.velocity - robot1.velocity
        
        # Minimum distance threshold
        min_dist = robot1.radius + robot2.radius + self.collision_threshold
        
        # Check if robots are already too close
        current_dist = np.linalg.norm(rel_pos)
        if current_dist < min_dist:
            return 0.0  # Collision now
        
        # Check if robots are moving apart
        if np.dot(rel_pos, rel_vel) >= 0:
            return None  # No collision
        
        # Calculate collision time using quadratic formula
        a = np.dot(rel_vel, rel_vel)
        if a == 0:
            return None
        
        b = 2 * np.dot(rel_pos, rel_vel)
        c = np.dot(rel_pos, rel_pos) - min_dist**2
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None
        
        t = (-b - np.sqrt(discriminant)) / (2*a)
        
        if t < 0 or t > self.prediction_horizon:
            return None
        
        return t
    
    def _calculate_avoidance_velocity(self, robot: Robot, obstacles: List[Robot]) -> np.ndarray:
        """Calculate collision avoidance velocity"""

        avoidance = np.zeros(2)
        
        for obstacle in obstacles:
            if obstacle.id == robot.id:
                continue
            
            collision_time = self._predict_collision(robot, obstacle)
            if collision_time is not None and collision_time < self.prediction_horizon:
                # Calculate avoidance direction (perpendicular to collision direction)
                rel_pos = obstacle.position - robot.position
                distance = np.linalg.norm(rel_pos)
                
                if distance > 0:
                    # Avoidance force inversely proportional to time to collision
                    force = 1.0 / (collision_time + 0.1)
                    direction = -rel_pos / distance
                    avoidance += force * direction
        
        return avoidance
    
    def _tracking_loop(self):
        """Main tracking loop"""

        while self.running:
            try:
                # Collect frames from all cameras
                all_detections = defaultdict(list)
                
                for serial, pipeline in self.pipelines.items():
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    
                    if not color_frame or not depth_frame:
                        continue
                    
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    camera_config = next(c for c in self.camera_configs if c.serial == serial)
                    
                    # Detect robots
                    for robot_id, robot in self.robots.items():
                        points = self._detect_colored_points(color_image, robot.color_hsv_range)
                        
                        for point in points:
                            # Get depth at point
                            depth = depth_image[int(point[1]), int(point[0])]
                            if depth > 0:
                                world_point = self._pixel_to_world(point, camera_config, depth / 1000.0)
                                all_detections[robot_id].append(world_point)
                
                # Update robot positions
                for robot_id, robot in self.robots.items():
                    self._update_robot_position(robot, all_detections[robot_id])
                
                # Check for collisions and calculate avoidance
                robots_list = list(self.robots.values())
                for robot in robots_list:
                    avoidance_vel = self._calculate_avoidance_velocity(robot, robots_list)
                    # Store avoidance suggestion (to be sent via ZMQ)
                    robot.avoidance_velocity = avoidance_vel
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in tracking loop: {e}")
    
    def _zmq_loop(self):
        """Handle ZMQ requests"""

        while self.running:
            try:
                # Check for messages with timeout
                if self.socket.poll(100):
                    message = self.socket.recv_json()
                    
                    if message['type'] == 'get_state':
                        robot_id = message['robot_id']
                        if robot_id in self.robots:
                            robot = self.robots[robot_id]
                            response = {
                                'position': robot.position.tolist(),
                                'velocity': robot.velocity.tolist(),
                                'avoidance_velocity': getattr(robot, 'avoidance_velocity', np.zeros(2)).tolist(),
                                'timestamp': robot.last_update
                            }
                        else:
                            response = {'error': 'Robot not found'}
                    
                    elif message['type'] == 'get_all_robots':
                        response = {
                            robot_id: {
                                'position': robot.position.tolist(),
                                'velocity': robot.velocity.tolist()
                            }
                            for robot_id, robot in self.robots.items()
                        }
                    
                    else:
                        response = {'error': 'Unknown message type'}
                    
                    self.socket.send_json(response)
                    
            except Exception as e:
                logger.error(f"Error in ZMQ loop: {e}")
    
    def start(self):
        """Start tracking"""

        self.running = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.zmq_thread = threading.Thread(target=self._zmq_loop)
        
        self.tracking_thread.start()
        self.zmq_thread.start()
        
        logger.info("Tracker started")
    
    def stop(self):
        """Stop tracking"""

        self.running = False
        
        if self.tracking_thread:
            self.tracking_thread.join()
        if self.zmq_thread:
            self.zmq_thread.join()
        
        # Close cameras
        for pipeline in self.pipelines.values():
            pipeline.stop()
        
        # Close ZMQ
        self.socket.close()
        self.context.term()
        
        logger.info("Tracker stopped")


class RobotClient:
    """Example robot client that communicates with the tracker"""
    
    def __init__(self, robot_id: str, tracker_address: str = "tcp://localhost:5555"):
        self.robot_id = robot_id
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(tracker_address)
    
    def get_state(self) -> Dict:
        """Get current robot state from tracker"""
        self.socket.send_json({
            'type': 'get_state',
            'robot_id': self.robot_id
        })
        return self.socket.recv_json()
    
    def get_all_robots(self) -> Dict:
        """Get all robots' positions"""
        self.socket.send_json({'type': 'get_all_robots'})
        return self.socket.recv_json()
    
    def close(self):
        """Close connection"""
        self.socket.close()
        self.context.term()


# Example usage
if __name__ == "__main__":
    # Define camera configurations
    # You'll need to adjust these based on your actual camera setup
    camera_configs = [
        CameraConfig(
            serial="123456",  # Replace with actual serial
            position=np.array([0, 0, 1]),  # 1 meter high
            rotation=np.eye(3)  # No rotation
        ),
        CameraConfig(
            serial="789012",  # Replace with actual serial
            position=np.array([2, 0, 1]),  # 2 meters to the right
            rotation=np.eye(3)
        )
    ]
    
    # Create tracker
    tracker = MultiCameraTracker(camera_configs)
    
    # Register robots with their color ranges (HSV)
    # Example: red robot
    tracker.register_robot(
        "robot_1",
        color_hsv_range=(
            np.array([0, 100, 100]),    # Lower HSV bound
            np.array([10, 255, 255])    # Upper HSV bound
        ),
        radius=25
    )
    
    # Example: blue robot
    tracker.register_robot(
        "robot_2",
        color_hsv_range=(
            np.array([100, 100, 100]),
            np.array([130, 255, 255])
        ),
        radius=25
    )
    
    try:
        # Start tracking
        tracker.start()
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        tracker.stop()