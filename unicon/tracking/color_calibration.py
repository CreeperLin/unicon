#!/usr/bin/env python3
"""
Color calibration tool for finding HSV ranges of robot markers
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import json


class ColorCalibrator:
    def __init__(self):
        self.hsv_lower = np.array([0, 0, 0])
        self.hsv_upper = np.array([180, 255, 255])
        self.window_name = "Color Calibration"
        self.mask_window = "Mask Preview"
        self.calibrations = {}
        
    def nothing(self, x):
        """Callback for trackbar"""
        pass
    
    def create_trackbars(self):
        """Create HSV trackbars"""
        cv2.namedWindow(self.window_name)
        cv2.namedWindow(self.mask_window)
        
        # Create trackbars for HSV values
        cv2.createTrackbar('H Min', self.window_name, 0, 180, self.nothing)
        cv2.createTrackbar('S Min', self.window_name, 0, 255, self.nothing)
        cv2.createTrackbar('V Min', self.window_name, 0, 255, self.nothing)
        cv2.createTrackbar('H Max', self.window_name, 180, 180, self.nothing)
        cv2.createTrackbar('S Max', self.window_name, 255, 255, self.nothing)
        cv2.createTrackbar('V Max', self.window_name, 255, 255, self.nothing)
        
        # Morphology trackbars
        cv2.createTrackbar('Erode', self.window_name, 0, 10, self.nothing)
        cv2.createTrackbar('Dilate', self.window_name, 0, 10, self.nothing)
    
    def get_trackbar_values(self):
        """Get current trackbar values"""
        h_min = cv2.getTrackbarPos('H Min', self.window_name)
        s_min = cv2.getTrackbarPos('S Min', self.window_name)
        v_min = cv2.getTrackbarPos('V Min', self.window_name)
        h_max = cv2.getTrackbarPos('H Max', self.window_name)
        s_max = cv2.getTrackbarPos('S Max', self.window_name)
        v_max = cv2.getTrackbarPos('V Max', self.window_name)
        
        erode = cv2.getTrackbarPos('Erode', self.window_name)
        dilate = cv2.getTrackbarPos('Dilate', self.window_name)
        
        return (np.array([h_min, s_min, v_min]), 
                np.array([h_max, s_max, v_max]),
                erode, dilate)
    
    def calibrate_with_camera(self, camera_serial: str = None):
        """Calibrate colors using RealSense camera"""
        # Configure RealSense
        pipeline = rs.pipeline()
        config = rs.config()
        
        if camera_serial:
            config.enable_device(camera_serial)
        
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        pipeline.start(config)
        
        # Create trackbars
        self.create_trackbars()
        
        print("Color Calibration Tool")
        print("---------------------")
        print("Instructions:")
        print("1. Adjust HSV sliders to isolate your robot marker color")
        print("2. Press 'S' to save current color calibration")
        print("3. Press 'L' to load saved calibrations")
        print("4. Press 'ESC' to exit")
        print("5. Click on the image to sample color at that point")
        
        # Mouse callback for color sampling
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                hsv_pixel = hsv_image[y, x]
                print(f"HSV at ({x}, {y}): {hsv_pixel}")
                
                # Set trackbar values around sampled color
                margin = 20
                cv2.setTrackbarPos('H Min', self.window_name, max(0, hsv_pixel[0] - margin))
                cv2.setTrackbarPos('H Max', self.window_name, min(180, hsv_pixel[0] + margin))
                cv2.setTrackbarPos('S Min', self.window_name, max(0, hsv_pixel[1] - 50))
                cv2.setTrackbarPos('S Max', self.window_name, 255)
                cv2.setTrackbarPos('V Min', self.window_name, max(0, hsv_pixel[2] - 50))
                cv2.setTrackbarPos('V Max', self.window_name, 255)
        
        cv2.setMouseCallback(self.window_name, mouse_callback)
        
        try:
            while True:
                # Get frame
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
                
                # Get trackbar values
                hsv_lower, hsv_upper, erode_size, dilate_size = self.get_trackbar_values()
                
                # Create mask
                mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
                
                # Apply morphology
                if erode_size > 0:
                    kernel = np.ones((erode_size, erode_size), np.uint8)
                    mask = cv2.erode(mask, kernel, iterations=1)
                
                if dilate_size > 0:
                    kernel = np.ones((dilate_size, dilate_size), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours and centers
                result = color_image.copy()
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Minimum area threshold
                        # Draw contour
                        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
                        
                        # Calculate and draw center
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
                            cv2.putText(result, f"Area: {int(area)}", (cx-30, cy-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add HSV values to display
                cv2.putText(result, f"HSV: [{hsv_lower[0]}-{hsv_upper[0]}, "
                                  f"{hsv_lower[1]}-{hsv_upper[1]}, "
                                  f"{hsv_lower[2]}-{hsv_upper[2]}]",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show images
                cv2.imshow(self.window_name, result)
                cv2.imshow(self.mask_window, mask)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                
                elif key == ord('s'):  # Save calibration
                    robot_name = input("Enter robot name: ")
                    self.calibrations[robot_name] = {
                        'hsv_lower': hsv_lower.tolist(),
                        'hsv_upper': hsv_upper.tolist(),
                        'erode': erode_size,
                        'dilate': dilate_size
                    }
                    print(f"Saved calibration for {robot_name}")
                    print(f"HSV Range: {hsv_lower} - {hsv_upper}")
                
                elif key == ord('l'):  # Load calibrations
                    self.load_calibrations()
                    print("Available calibrations:")
                    for name, calib in self.calibrations.items():
                        print(f"  {name}: {calib['hsv_lower']} - {calib['hsv_upper']}")
        
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
    
    def calibrate_with_image(self, image_path: str):
        """Calibrate colors using a static image"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create trackbars
        self.create_trackbars()
        
        print("Calibrating with static image")
        print("Press 'S' to save, 'ESC' to exit")
        
        while True:
            # Get trackbar values
            hsv_lower, hsv_upper, erode_size, dilate_size = self.get_trackbar_values()
            
            # Create mask
            mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
            
            # Apply morphology
            if erode_size > 0:
                kernel = np.ones((erode_size, erode_size), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
            
            if dilate_size > 0:
                kernel = np.ones((dilate_size, dilate_size), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Apply mask
            result = cv2.bitwise_and(image, image, mask=mask)
            
            # Show images
            cv2.imshow(self.window_name, result)
            cv2.imshow(self.mask_window, mask)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):
                robot_name = input("Enter robot name: ")
                self.calibrations[robot_name] = {
                    'hsv_lower': hsv_lower.tolist(),
                    'hsv_upper': hsv_upper.tolist(),
                    'erode': erode_size,
                    'dilate': dilate_size
                }
                print(f"Saved calibration for {robot_name}")
        
        cv2.destroyAllWindows()
    
    def save_calibrations(self, filename: str = "color_calibrations.json"):
        """Save all calibrations to file"""
        with open(filename, 'w') as f:
            json.dump(self.calibrations, f, indent=2)
        print(f"Saved calibrations to {filename}")
    
    def load_calibrations(self, filename: str = "color_calibrations.json"):
        """Load calibrations from file"""
        try:
            with open(filename, 'r') as f:
                self.calibrations = json.load(f)
            print(f"Loaded calibrations from {filename}")
        except FileNotFoundError:
            print(f"No calibration file found: {filename}")


def main():
    """Main calibration program"""
    calibrator = ColorCalibrator()
    
    print("Color Calibration Tool")
    print("1. Calibrate with camera")
    print("2. Calibrate with image")
    print("3. Exit")
    
    choice = input("Select option: ")
    
    if choice == '1':
        # List available cameras
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("No RealSense cameras found!")
            return
        
        print("\nAvailable cameras:")
        serials = []
        for i, device in enumerate(devices):
            serial = device.get_info(rs.camera_info.serial_number)
            serials.append(serial)
            print(f"{i+1}. {serial}")
        
        if len(serials) == 1:
            calibrator.calibrate_with_camera(serials[0])
        else:
            idx = int(input("Select camera (number): ")) - 1
            calibrator.calibrate_with_camera(serials[idx])
    
    elif choice == '2':
        image_path = input("Enter image path: ")
        calibrator.calibrate_with_image(image_path)
    
    # Save calibrations
    if calibrator.calibrations:
        calibrator.save_calibrations()
        
        print("\nCalibration summary:")
        for name, calib in calibrator.calibrations.items():
            print(f"\n{name}:")
            print(f"  HSV Lower: {calib['hsv_lower']}")
            print(f"  HSV Upper: {calib['hsv_upper']}")


if __name__ == "__main__":
    main()