import zmq
import json
import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Callable
import logging


@dataclass
class Position:
    x: int
    y: int
    confidence: float
    camera_id: int
    timestamp: float


class RobotClient:
    def __init__(self, robot_id: str, tracker_host: str = "localhost", tracker_port: int = 5555):
        self.robot_id = robot_id
        self.tracker_host = tracker_host
        self.tracker_port = tracker_port
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{tracker_host}:{tracker_port}")
        
        # Subscribe to relevant topics
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "positions")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "warnings")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, f"command_{robot_id}")
        
        # State
        self.current_position: Optional[Position] = None
        self.other_robots: Dict[str, Position] = {}
        self.is_running = False
        self.collision_warnings = []
        
        # Callbacks
        self.position_callback: Optional[Callable] = None
        self.collision_callback: Optional[Callable] = None
        self.command_callback: Optional[Callable] = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"Robot_{robot_id}")
        
        self.logger.info(f"Robot {robot_id} initialized, connecting to tracker at {tracker_host}:{tracker_port}")
    
    def set_position_callback(self, callback: Callable[[Position], None]):
        """Set callback for when robot position is updated"""
        self.position_callback = callback
    
    def set_collision_callback(self, callback: Callable[[list], None]):
        """Set callback for collision warnings"""
        self.collision_callback = callback
    
    def set_command_callback(self, callback: Callable[[dict], None]):
        """Set callback for commands from tracker"""
        self.command_callback = callback
    
    def start_listening(self):
        """Start listening for messages from tracker"""
        self.is_running = True
        listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        listener_thread.start()
        self.logger.info("Started listening for tracker messages")
    
    def _listen_loop(self):
        """Main listening loop"""
        while self.is_running:
            try:
                # Non-blocking receive with timeout
                topic, message = self.socket.recv_multipart(zmq.NOBLOCK)
                topic = topic.decode('utf-8')
                message_data = json.loads(message.decode('utf-8'))
                
                self._process_message(topic, message_data)
                
            except zmq.Again:
                # No message available, continue
                time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Error in listen loop: {e}")
                time.sleep(0.1)
    
    def _process_message(self, topic: str, message_data: dict):
        """Process received message based on topic"""
        try:
            if topic == "positions":
                self._handle_positions(message_data)
            elif topic == "warnings":
                self._handle_warnings(message_data)
            elif topic.startswith(f"command_{self.robot_id}"):
                self._handle_command(message_data)
                
        except Exception as e:
            self.logger.error(f"Error processing message {topic}: {e}")
    
    def _handle_positions(self, message_data: dict):
        """Handle position updates"""
        positions_data = message_data.get("data", {})
        
        # Update our own position
        if self.robot_id in positions_data:
            pos_data = positions_data[self.robot_id]
            self.current_position = Position(
                x=pos_data["x"],
                y=pos_data["y"],
                confidence=pos_data["confidence"],
                camera_id=pos_data["camera_id"],
                timestamp=message_data["timestamp"]
            )
            
            if self.position_callback:
                self.position_callback(self.current_position)
        
        # Update other robots' positions
        self.other_robots.clear()
        for robot_id, pos_data in positions_data.items():
            if robot_id != self.robot_id:
                self.other_robots[robot_id] = Position(
                    x=pos_data["x"],
                    y=pos_data["y"],
                    confidence=pos_data["confidence"],
                    camera_id=pos_data["camera_id"],
                    timestamp=message_data["timestamp"]
                )
    
    def _handle_warnings(self, message_data: dict):
        """Handle collision warnings"""
        warnings = message_data.get("warnings", [])
        
        # Filter warnings relevant to this robot
        relevant_warnings = [
            w for w in warnings 
            if w["robot1"] == self.robot_id or w["robot2"] == self.robot_id
        ]
        
        if relevant_warnings:
            self.collision_warnings = relevant_warnings
            self.logger.warning(f"Received {len(relevant_warnings)} collision warnings")
            
            if self.collision_callback:
                self.collision_callback(relevant_warnings)
    
    def _handle_command(self, message_data: dict):
        """Handle commands from tracker"""
        command = message_data.get("command", {})
        self.logger.info(f"Received command: {command}")
        
        if self.command_callback:
            self.command_callback(command)
        else:
            # Default command handling
            self._execute_default_command(command)
    
    def _execute_default_command(self, command: dict):
        """Default command execution"""
        action = command.get("action")
        
        if action == "stop":
            self.logger.warning("EMERGENCY STOP - Collision imminent!")
            # Implement emergency stop logic here
            
        elif action == "slow_down":
            factor = command.get("factor", 0.5)
            self.logger.info(f"Slowing down by factor {factor}")
            # Implement speed reduction logic here
            
        elif action == "change_path":
            self.logger.info("Changing path to avoid collision")
            # Implement path change logic here
    
    def get_position(self) -> Optional[Position]:
        """Get current robot position"""
        return self.current_position
    
    def get_other_robots(self) -> Dict[str, Position]:
        """Get positions of other robots"""
        return self.other_robots.copy()
    
    def get_collision_warnings(self) -> list:
        """Get current collision warnings"""
        return self.collision_warnings.copy()
    
    def is_collision_imminent(self, threshold_distance: float = 50.0) -> bool:
        """Check if collision is imminent"""
        for warning in self.collision_warnings:
            if warning["distance"] < threshold_distance:
                return True
        return False
    
    def stop(self):
        """Stop the robot client"""
        self.is_running = False
        self.socket.close()
        self.context.term()
        self.logger.info("Robot client stopped")


# Example robot implementation
class ExampleRobot:
    def __init__(self, robot_id: str):
        self.robot_id = robot_id
        self.client = RobotClient(robot_id)
        self.speed = 1.0
        self.is_stopped = False
        
        # Set up callbacks
        self.client.set_position_callback(self.on_position_update)
        self.client.set_collision_callback(self.on_collision_warning)
        self.client.set_command_callback(self.on_command_received)
        
        self.logger = logging.getLogger(f"ExampleRobot_{robot_id}")
    
    def on_position_update(self, position: Position):
        """Called when robot position is updated"""
        self.logger.info(f"Position: ({position.x}, {position.y}) confidence: {position.confidence:.2f}")
    
    def on_collision_warning(self, warnings: list):
        """Called when collision warnings are received"""
        for warning in warnings:
            other_robot = warning["robot2"] if warning["robot1"] == self.robot_id else warning["robot1"]
            distance = warning["distance"]
            self.logger.warning(f"Collision warning with {other_robot}, distance: {distance:.1f}px")
    
    def on_command_received(self, command: dict):
        """Called when command is received from tracker"""
        action = command.get("action")
        
        if action == "stop":
            self.emergency_stop()
        elif action == "slow_down":
            factor = command.get("factor", 0.5)
            self.set_speed(self.speed * factor)
        
        self.logger.info(f"Executed command: {command}")
    
    def emergency_stop(self):
        """Emergency stop the robot"""
        self.is_stopped = True
        self.speed = 0.0
        self.logger.warning("EMERGENCY STOP ACTIVATED!")
    
    def set_speed(self, speed: float):
        """Set robot speed"""
        self.speed = max(0.0, min(speed, 1.0))  # Clamp between 0 and 1
        self.logger.info(f"Speed set to {self.speed:.2f}")
    
    def start(self):
        """Start the robot"""
        self.client.start_listening()
        self.logger.info(f"Robot {self.robot_id} started")
        
        # Example movement loop
        try:
            while True:
                if not self.is_stopped:
                    # Get current position and other robots
                    my_pos = self.client.get_position()
                    other_robots = self.client.get_other_robots()
                    
                    if my_pos:
                        self.logger.info(f"My position: ({my_pos.x}, {my_pos.y}), "
                                       f"Other robots: {len(other_robots)}, "
                                       f"Speed: {self.speed:.2f}")
                    
                    # Check for imminent collision
                    if self.client.is_collision_imminent(60.0):
                        if not self.is_stopped:
                            self.logger.warning("Collision imminent - reducing speed")
                            self.set_speed(self.speed * 0.5)
                
                time.sleep(1.0)  # Update every second
                
        except KeyboardInterrupt:
            self.logger.info("Robot stopped by user")
        finally:
            self.client.stop()


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        robot_id = sys.argv[1]
    else:
        robot_id = "robot_red"  # Default robot ID
    
    # Create and start robot
    robot = ExampleRobot(robot_id)
    
    print(f"Starting robot {robot_id}")
    print("Make sure the tracker is running and this robot's color is configured")
    print("Press Ctrl+C to stop")
    
    robot.start()