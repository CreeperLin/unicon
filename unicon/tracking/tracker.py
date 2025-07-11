import cv2
import numpy as np
import pyrealsense2 as rs
import zmq

import time
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
import traceback

@dataclass
class RobotPosition:
    x: int
    y: int
    color: str
    confidence: float
    timestamp: float
    camera_id: int

@dataclass
class CollisionWarning:
    robot1: str
    robot2: str
    distance: float
    safe_distance: int


class ColorDetector:
    def __init__(self):
        self.reference_colors = {}
        self.color_ranges = {}
        
    def add_reference_color(self, color_name: str, image_path: str, sample_points: List[Tuple[int, int]] = None):
        """Add a reference color from an image sample"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        if sample_points:
            # Use specific sample points
            color_samples = [hsv[y, x] for x, y in sample_points]
        else:
            # Use center region as sample
            h, w = hsv.shape[:2]
            center_region = hsv[h//4:3*h//4, w//4:3*w//4]
            color_samples = center_region.reshape(-1, 3)
        
        # Calculate color statistics
        color_samples = np.array(color_samples)
        mean_color = np.mean(color_samples, axis=0)
        std_color = np.std(color_samples, axis=0)
        
        # Create HSV range (adjust tolerance as needed)
        tolerance = [20, 50, 50]  # H, S, V tolerance
        lower = np.maximum(mean_color - std_color - tolerance, [0, 0, 0])
        upper = np.minimum(mean_color + std_color + tolerance, [179, 255, 255])
        
        self.reference_colors[color_name] = mean_color
        self.color_ranges[color_name] = (lower.astype(np.uint8), upper.astype(np.uint8))
        
        logging.info(f"Added color '{color_name}': HSV={mean_color}, Range={self.color_ranges[color_name]}")
    
    def detect_colors(self, image: np.ndarray) -> Dict[str, List[Tuple[int, int, float]]]:
        """Detect all reference colors in the image"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detections = {}
        
        for color_name, (lower, upper) in self.color_ranges.items():
            # Create mask for this color
            mask = cv2.inRange(hsv, lower, upper)
            
            # Apply morphological operations to clean up noise
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            color_detections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    # Calculate centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Calculate confidence based on area and circularity
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                        confidence = min(area / 1000.0, 1.0) * circularity
                        
                        color_detections.append((cx, cy, confidence))
            
            # Sort by confidence and take the best detection
            if color_detections:
                color_detections.sort(key=lambda x: x[2], reverse=True)
                detections[color_name] = color_detections[:1]  # Take only the best detection
        
        return detections


class CameraManager:
    def __init__(self):
        self.pipelines = []
        self.camera_configs = []
        self.camera_matrices = []
        self.intrinsics = []
        
    def initialize_cameras(self) -> int:
        """Initialize all available RealSense cameras"""
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            logging.warning("No RealSense devices found, using webcams instead")
            return self._initialize_webcams()
        
        for i, device in enumerate(devices):
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                
                # Enable color stream
                config.enable_device(device.get_info(rs.camera_info.serial_number))
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                
                # Start pipeline
                profile = pipeline.start(config)
                
                # Get camera intrinsics for coordinate transformation
                color_profile = profile.get_stream(rs.stream.color)
                intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
                
                self.pipelines.append(pipeline)
                self.camera_configs.append(config)
                self.intrinsics.append(intrinsics)
                
                logging.info(f"Initialized RealSense camera {i}")
                
            except Exception as e:
                logging.error(f"Failed to initialize camera {i}: {e}")
        
        return len(self.pipelines)
    
    def _initialize_webcams(self) -> int:
        """Fallback to webcams if no RealSense cameras available"""
        camera_count = 0
        for i in range(4):  # Try up to 4 cameras
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.pipelines.append(cap)
                camera_count += 1
                logging.info(f"Initialized webcam {i}")
            else:
                cap.release()
        
        return camera_count
    
    def get_frames(self) -> List[Optional[np.ndarray]]:
        """Get frames from all cameras"""
        frames = []
        
        for i, pipeline in enumerate(self.pipelines):
            try:
                if isinstance(pipeline, rs.pipeline):
                    # RealSense camera
                    frameset = pipeline.wait_for_frames(timeout_ms=1000)
                    color_frame = frameset.get_color_frame()
                    if color_frame:
                        frame = np.asanyarray(color_frame.get_data())
                        frames.append(frame)
                    else:
                        frames.append(None)
                else:
                    # Webcam
                    ret, frame = pipeline.read()
                    frames.append(frame if ret else None)
                    
            except Exception as e:
                logging.error(f"Error getting frame from camera {i}: {e}")
                frames.append(None)
        
        return frames
    
    def release_cameras(self):
        """Release all camera resources"""
        for pipeline in self.pipelines:
            try:
                if isinstance(pipeline, rs.pipeline):
                    pipeline.stop()
                else:
                    pipeline.release()
            except Exception as e:
                logging.error(f"Error releasing camera: {e}")


class ImageStitcher:
    def __init__(self, overlap_threshold: float = 0.1):
        self.overlap_threshold = overlap_threshold
        self.transform_matrices = []
        self.is_calibrated = False
        
    def calibrate_cameras(self, frames: List[np.ndarray]):
        """Calibrate camera positions using feature matching"""
        if len(frames) < 2:
            return
            
        # For simplicity, arrange cameras in a grid layout
        # In a real implementation, you'd use feature matching or markers
        self.is_calibrated = True
        logging.info("Camera calibration completed (simplified)")
    
    def stitch_images(self, frames: List[np.ndarray]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Stitch multiple camera images into one combined view"""
        valid_frames = [f for f in frames if f is not None]
        
        if not valid_frames:
            return None, []
        
        if len(valid_frames) == 1:
            return valid_frames[0], [(0, 0)]
        
        # Simple grid arrangement for multiple cameras
        # For a more sophisticated approach, use OpenCV's stitching algorithm
        rows = int(np.ceil(np.sqrt(len(valid_frames))))
        cols = int(np.ceil(len(valid_frames) / rows))
        
        if valid_frames:
            h, w = valid_frames[0].shape[:2]
            stitched = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
            offsets = []
            
            for i, frame in enumerate(valid_frames):
                row = i // cols
                col = i % cols
                y_offset = row * h
                x_offset = col * w
                stitched[y_offset:y_offset + h, x_offset:x_offset + w] = frame
                offsets.append((x_offset, y_offset))
            
            return stitched, offsets
        
        return None, []


class CollisionAvoidance:
    def __init__(self, safe_distance: int = 50, warning_distance: int = 100):
        self.safe_distance = safe_distance
        self.warning_distance = warning_distance
        
    def check_collisions(self, robot_positions: Dict[str, RobotPosition]) -> List[CollisionWarning]:
        """Check for potential collisions between robots"""
        warnings = []
        robots = list(robot_positions.keys())
        
        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                robot1, robot2 = robots[i], robots[j]
                pos1, pos2 = robot_positions[robot1], robot_positions[robot2]
                
                distance = np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)
                
                if distance < self.warning_distance:
                    warnings.append(CollisionWarning(
                        robot1=robot1,
                        robot2=robot2,
                        distance=distance,
                        safe_distance=self.safe_distance
                    ))
        
        return warnings
    
    def generate_avoidance_commands(self, warnings: List[CollisionWarning], 
                                  robot_positions: Dict[str, RobotPosition]) -> Dict[str, Dict]:
        """Generate avoidance commands for robots"""
        commands = {}
        
        for warning in warnings:
            if warning.distance < self.safe_distance:
                # Emergency stop
                commands[warning.robot1] = {"action": "stop", "reason": "collision_imminent"}
                commands[warning.robot2] = {"action": "stop", "reason": "collision_imminent"}
            elif warning.distance < self.warning_distance:
                # Slow down and adjust path
                commands[warning.robot1] = {"action": "slow_down", "factor": 0.5}
                commands[warning.robot2] = {"action": "slow_down", "factor": 0.5}
        
        return commands


class ZMQCommunicator:
    def __init__(self, port: int = 5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        self.port = port
        logging.info(f"ZMQ Publisher started on port {port}")
    
    def send_robot_positions(self, positions: Dict[str, RobotPosition]):
        """Send robot positions to all connected robots"""
        message = {
            "type": "positions",
            "timestamp": time.time(),
            "data": {
                robot_id: {
                    "x": pos.x,
                    "y": pos.y,
                    "confidence": pos.confidence,
                    "camera_id": pos.camera_id
                }
                for robot_id, pos in positions.items()
            }
        }
        self.socket.send_string(f"positions {json.dumps(message)}")
    
    def send_collision_warnings(self, warnings: List[CollisionWarning]):
        """Send collision warnings"""
        message = {
            "type": "collision_warnings",
            "timestamp": time.time(),
            "warnings": [
                {
                    "robot1": w.robot1,
                    "robot2": w.robot2,
                    "distance": w.distance,
                    "safe_distance": w.safe_distance
                }
                for w in warnings
            ]
        }
        self.socket.send_string(f"warnings {json.dumps(message)}")
    
    def send_commands(self, commands: Dict[str, Dict]):
        """Send avoidance commands to specific robots"""
        for robot_id, command in commands.items():
            message = {
                "type": "command",
                "timestamp": time.time(),
                "robot_id": robot_id,
                "command": command
            }
            self.socket.send_string(f"command_{robot_id} {json.dumps(message)}")
    
    def close(self):
        self.socket.close()
        self.context.term()


class Tracker:
    def __init__(self, safe_distance: int = 50, max_history: int = 10):
        self.color_detector = ColorDetector()
        self.camera_manager = CameraManager()
        self.image_stitcher = ImageStitcher()
        self.collision_avoidance = CollisionAvoidance(safe_distance=safe_distance)
        self.zmq_communicator = ZMQCommunicator()
        
        self.robot_positions = {}
        self.position_history = defaultdict(lambda: deque(maxlen=max_history))
        self.running = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_robot_color(self, robot_id: str, image_path: str, sample_points: List[Tuple[int, int]] = None):
        """Add a robot's reference color from an image"""
        try:
            self.color_detector.add_reference_color(robot_id, image_path, sample_points)
            self.logger.info(f"Added robot {robot_id} with reference color from {image_path}")
        except Exception as e:
            self.logger.error(f"Failed to add robot color for {robot_id}: {e}")
    
    def initialize(self) -> bool:
        """Initialize the tracking system"""
        try:
            camera_count = self.camera_manager.initialize_cameras()
            if camera_count == 0:
                self.logger.error("No cameras initialized")
                return False
            
            self.logger.info(f"Initialized {camera_count} cameras")
            
            # Calibrate camera positions
            frames = self.camera_manager.get_frames()
            valid_frames = [f for f in frames if f is not None]
            if valid_frames:
                self.image_stitcher.calibrate_cameras(valid_frames)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def process_frame(self):
        """Process a single frame from all cameras"""
        frames = self.camera_manager.get_frames()
        
        is_valid = [e is not None for e in frames]
        if not any(is_valid):
            return None, {}
        
        # Stitch images together
        stitched_image, camera_offsets = self.image_stitcher.stitch_images(frames)
        
        if stitched_image is None:
            return None, {}
        
        # Detect robots in the stitched image
        detections = self.color_detector.detect_colors(stitched_image)
        
        # Update robot positions
        current_positions = {}
        current_time = time.time()
        
        for robot_id, robot_detections in detections.items():
            if robot_detections:
                x, y, confidence = robot_detections[0]  # Take the best detection
                
                # Determine which camera detected this robot
                camera_id = 0
                for i, (offset_x, offset_y) in enumerate(camera_offsets):
                    if len(frames) > i and frames[i] is not None:
                        h, w = frames[i].shape[:2]
                        if offset_x <= x < offset_x + w and offset_y <= y < offset_y + h:
                            camera_id = i
                            break
                
                position = RobotPosition(
                    x=x, y=y, color=robot_id, 
                    confidence=confidence, 
                    timestamp=current_time,
                    camera_id=camera_id
                )
                
                current_positions[robot_id] = position
                self.position_history[robot_id].append(position)
        
        self.robot_positions = current_positions
        return stitched_image, current_positions
    
    def run_tracking(self, display: bool = True):
        """Main tracking loop"""
        self.running = True
        
        while self.running:
            try:
                # Process current frame
                stitched_image, positions = self.process_frame()
                
                if stitched_image is not None:
                    # Check for collisions
                    warnings = self.collision_avoidance.check_collisions(positions)
                    
                    # Generate avoidance commands
                    commands = self.collision_avoidance.generate_avoidance_commands(warnings, positions)
                    
                    # Send data via ZMQ
                    if positions:
                        self.zmq_communicator.send_robot_positions(positions)
                    
                    if warnings:
                        self.zmq_communicator.send_collision_warnings(warnings)
                        self.logger.warning(f"Collision warnings: {len(warnings)}")
                    
                    if commands:
                        self.zmq_communicator.send_commands(commands)
                        self.logger.info(f"Sent commands to {len(commands)} robots")
                    
                    # Display results
                    if display:
                        self.display_results(stitched_image, positions, warnings)
                
                # Control frame rate
                time.sleep(0.033)  # ~30 FPS
                
            except KeyboardInterrupt:
                self.logger.info("Tracking stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in tracking loop: {e}")
                print(traceback.format_exc())
    
    def display_results(self, image: np.ndarray, positions: Dict[str, RobotPosition], 
                       warnings: List[CollisionWarning]):
        """Display tracking results"""
        display_image = image.copy()
        
        # Draw robot positions
        for robot_id, pos in positions.items():
            color = self._get_display_color(robot_id)
            cv2.circle(display_image, (pos.x, pos.y), 10, color, -1)
            cv2.putText(display_image, f"{robot_id} conf: ({pos.confidence:.2f})", 
                       (pos.x + 15, pos.y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw collision warnings
        for warning in warnings:
            if warning.robot1 in positions and warning.robot2 in positions:
                pos1 = positions[warning.robot1]
                pos2 = positions[warning.robot2]
                
                color = (0, 0, 255) if warning.distance < self.collision_avoidance.safe_distance else (0, 165, 255)
                cv2.line(display_image, (pos1.x, pos1.y), (pos2.x, pos2.y), color, 2)
                
                mid_x = (pos1.x + pos2.x) // 2
                mid_y = (pos1.y + pos2.y) // 2
                cv2.putText(display_image, f"{warning.distance:.1f}px", 
                           (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imshow("Robot Tracker", display_image)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.running = False
    
    def _get_display_color(self, robot_id: str) -> Tuple[int, int, int]:
        """Get a unique color for displaying each robot"""
        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        return colors[hash(robot_id) % len(colors)]
    
    def stop(self):
        """Stop the tracking system"""
        self.running = False
        self.camera_manager.release_cameras()
        self.zmq_communicator.close()
        cv2.destroyAllWindows()
        self.logger.info("Tracking system stopped")


# Example usage
if __name__ == "__main__":
    # Create tracker instance
    tracker = Tracker(safe_distance=50)
    
    try:
        # Add robot colors from reference images
        # You need to provide actual image paths and optionally sample points
        tracker.add_robot_color("robot_red", "/home/ethan/Downloads/red.jpg")
        tracker.add_robot_color("robot_blue", "/home/ethan/Downloads/blue.jpg")
        # tracker.add_robot_color("robot_green", "green_robot_sample.jpg")
        
        # Initialize the system
        if tracker.initialize():
            print("Tracker initialized successfully")
            print("Starting tracking... Press 'q' in the display window to quit")
            
            # Start tracking
            tracker.run_tracking(display=True)
        else:
            print("Failed to initialize tracker")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        tracker.stop()