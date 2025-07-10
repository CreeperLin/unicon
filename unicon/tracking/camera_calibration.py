import cv2
import numpy as np
import pyrealsense2 as rs
import json
from typing import List, Tuple
import os

class CameraCalibrator:
    """Utility for calibrating multiple cameras and finding their relative positions"""
    
    def __init__(self, checkerboard_size: Tuple[int, int] = (9, 6), square_size: float = 0.025):
        """
        Initialize calibrator
        
        Args:
            checkerboard_size: Number of inner corners (columns, rows)
            square_size: Size of checkerboard squares in meters
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Prepare object points
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        self.cameras = {}
        self.calibration_data = {}
    
    def discover_cameras(self) -> List[str]:
        """Discover all connected RealSense cameras"""
        ctx = rs.context()
        devices = ctx.query_devices()
        
        serials = []
        for device in devices:
            serial = device.get_info(rs.camera_info.serial_number)
            serials.append(serial)
            print(f"Found camera: {serial}")
        
        return serials
    
    def capture_calibration_images(self, serial: str, num_images: int = 20):
        """Capture calibration images from a specific camera"""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Create directory for calibration images
        os.makedirs(f"calibration/{serial}", exist_ok=True)
        
        try:
            pipeline.start(config)
            
            print(f"Capturing calibration images for camera {serial}")
            print("Press SPACE to capture, ESC to finish")
            
            captured = 0
            while captured < num_images:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                image = np.asanyarray(color_frame.get_data())
                
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Find checkerboard corners
                ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
                
                # Draw corners
                display = image.copy()
                if ret:
                    cv2.drawChessboardCorners(display, self.checkerboard_size, corners, ret)
                
                cv2.imshow(f'Camera {serial} - Calibration', display)
                
                key = cv2.waitKey(1)
                if key == ord(' ') and ret:
                    # Save image
                    filename = f"calibration/{serial}/img_{captured:03d}.jpg"
                    cv2.imwrite(filename, image)
                    captured += 1
                    print(f"Captured {captured}/{num_images}")
                elif key == 27:  # ESC
                    break
            
            cv2.destroyAllWindows()
            pipeline.stop()
            
        except Exception as e:
            print(f"Error: {e}")
            pipeline.stop()
    
    def calibrate_camera(self, serial: str) -> dict:
        """Calibrate a single camera using captured images"""
        image_dir = f"calibration/{serial}"
        if not os.path.exists(image_dir):
            print(f"No calibration images found for camera {serial}")
            return None
        
        # Arrays to store object points and image points
        objpoints = []
        imgpoints = []
        
        images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        
        for fname in images:
            img = cv2.imread(os.path.join(image_dir, fname))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            if ret:
                objpoints.append(self.objp)
                
                # Refine corners
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners2)
        
        if len(objpoints) == 0:
            print(f"No valid calibration images for camera {serial}")
            return None
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        calibration = {
            'serial': serial,
            'camera_matrix': mtx.tolist(),
            'distortion': dist.tolist(),
            'calibration_error': ret
        }
        
        print(f"Camera {serial} calibrated with error: {ret}")
        
        return calibration
    
    def find_relative_poses(self, reference_serial: str, target_serials: List[str]):
        """Find relative poses between cameras using a shared checkerboard view"""
        print(f"\nFinding relative poses with reference camera {reference_serial}")
        print("Place checkerboard visible to all cameras and press SPACE")
        
        # Start all pipelines
        pipelines = {}
        for serial in [reference_serial] + target_serials:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            pipeline.start(config)
            pipelines[serial] = pipeline
        
        relative_poses = {}
        capturing = True
        
        while capturing:
            images = {}
            
            # Capture from all cameras
            for serial, pipeline in pipelines.items():
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if color_frame:
                    image = np.asanyarray(color_frame.get_data())
                    images[serial] = image
                    
                    # Show preview
                    cv2.imshow(f'Camera {serial}', cv2.resize(image, (320, 240)))
            
            key = cv2.waitKey(1)
            if key == ord(' '):
                # Process captured images
                corners_dict = {}
                
                for serial, image in images.items():
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
                    
                    if ret:
                        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                        corners_dict[serial] = corners2
                
                # Calculate relative poses
                if reference_serial in corners_dict:
                    ref_corners = corners_dict[reference_serial]
                    
                    for target_serial in target_serials:
                        if target_serial in corners_dict:
                            target_corners = corners_dict[target_serial]
                            
                            # Estimate relative pose
                            # This is simplified - in practice you'd use stereo calibration
                            relative_poses[target_serial] = self._estimate_relative_pose(
                                ref_corners, target_corners,
                                self.calibration_data[reference_serial],
                                self.calibration_data[target_serial]
                            )
                            
                            print(f"Found relative pose for camera {target_serial}")
                
                capturing = False
            
            elif key == 27:  # ESC
                break
        
        # Stop all pipelines
        for pipeline in pipelines.values():
            pipeline.stop()
        
        cv2.destroyAllWindows()
        
        return relative_poses
    
    def _estimate_relative_pose(self, corners1, corners2, calib1, calib2):
        """Estimate relative pose between two cameras"""
        # This is a simplified version - for production use stereoCalibrate
        mtx1 = np.array(calib1['camera_matrix'])
        dist1 = np.array(calib1['distortion'])
        mtx2 = np.array(calib2['camera_matrix'])
        dist2 = np.array(calib2['distortion'])
        
        # Find pose of checkerboard in each camera
        ret1, rvec1, tvec1 = cv2.solvePnP(self.objp, corners1, mtx1, dist1)
        ret2, rvec2, tvec2 = cv2.solvePnP(self.objp, corners2, mtx2, dist2)
        
        # Convert to rotation matrices
        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)
        
        # Relative rotation and translation
        R_rel = R2 @ R1.T
        t_rel = tvec2.ravel() - R_rel @ tvec1.ravel()
        
        return {
            'rotation': R_rel.tolist(),
            'translation': t_rel.tolist()
        }
    
    def save_calibration(self, filename: str = "camera_calibration.json"):
        """Save calibration data to file"""
        with open(filename, 'w') as f:
            json.dump(self.calibration_data, f, indent=2)
        print(f"Calibration saved to {filename}")
    
    def load_calibration(self, filename: str = "camera_calibration.json"):
        """Load calibration data from file"""
        with open(filename, 'r') as f:
            self.calibration_data = json.load(f)
        print(f"Calibration loaded from {filename}")


def main():
    """Main calibration workflow"""
    calibrator = CameraCalibrator()
    
    # Discover cameras
    serials = calibrator.discover_cameras()
    
    if len(serials) == 0:
        print("No cameras found!")
        return
    
    print(f"\nFound {len(serials)} cameras")
    
    # Calibrate each camera
    for serial in serials:
        print(f"\n--- Calibrating camera {serial} ---")
        
        # Capture calibration images
        calibrator.capture_calibration_images(serial, num_images=15)
        
        # Calibrate
        calib = calibrator.calibrate_camera(serial)
        if calib:
            calibrator.calibration_data[serial] = calib
    
    # Find relative poses
    if len(serials) > 1:
        reference = serials[0]
        targets = serials[1:]
        
        print(f"\n--- Finding relative poses ---")
        relative_poses = calibrator.find_relative_poses(reference, targets)
        
        # Add to calibration data
        calibrator.calibration_data['relative_poses'] = {
            'reference': reference,
            'poses': relative_poses
        }
    
    # Save calibration
    calibrator.save_calibration()
    
    print("\nCalibration complete!")


if __name__ == "__main__":
    main()