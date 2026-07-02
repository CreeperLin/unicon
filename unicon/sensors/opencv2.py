import cv2
import numpy as np
from typing import Callable, Optional, Union, Tuple


def cb_sensor_opencv2(
    states_color=None,
    source: Union[int, str] = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    resolution: Optional[str] = None,
    fps: int = 30,
    mode=None,
    auto_exposure: bool = True,
    exposure: Optional[float] = None,
    brightness: Optional[float] = None,
    contrast: Optional[float] = None,
    saturation: Optional[float] = None,
    hue: Optional[float] = None,
    gain: Optional[float] = None,
    fourcc: Optional[str] = None,
    backend: Optional[str] = None,
    verbose: bool = False,
    retry=False,
) -> Callable[[], bool]:

    capture = None
    
    if resolution and not width and not height:
        try:
            width_str, height_str = resolution.split('x')
            width = int(width_str)
            height = int(height_str)
            if verbose:
                print(f"cb_sensor_opencv2: Parsed resolution string to {width}x{height}")
        except ValueError:
            print(f"cb_sensor_opencv2: Invalid resolution format: {resolution}. Using defaults.")
    width = states_color.shape[1] if width is None else width
    height = states_color.shape[0] if height is None else height

    def init_cap() -> bool:
        nonlocal capture, width, height

        try:
            cap = cv2.CAP_ANY
            if backend is not None:
                cap = getattr(cv2, backend, cap)

            # Create capture object
            capture = cv2.VideoCapture(source, cap)

            print('cb_sensor_opencv2', source, capture.isOpened())

            if not capture.isOpened():
                print(f"cb_sensor_opencv2: Failed to open video source: {source} {cap} {cv2.__version__}")
                try:
                    import os
                    if isinstance(source, int):
                        device_path = f"/dev/video{source}"
                        print(f"cb_sensor_opencv2: Checking device path: {device_path}")
                        if os.path.exists(device_path):
                            print(f"cb_sensor_opencv2: Device file exists but OpenCV cannot open it")
                            # Check permissions
                            stat_info = os.stat(device_path)
                            print(f"cb_sensor_opencv2: Device permissions: {oct(stat_info.st_mode)[-3:]}")
                        else:
                            print(f"cb_sensor_opencv2: Device file does not exist: {device_path}")
                except Exception as cb_sensor_opencv2_e:
                    print(f"cb_sensor_opencv2: Error checking device: {cb_sensor_opencv2_e}")

                return False

            if width:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                actual_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                print(f"Set width to {width}, actual: {actual_width}")
            if height:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"Set height to {height}, actual: {actual_height}")
            if fps:
                capture.set(cv2.CAP_PROP_FPS, fps)
                actual_fps = capture.get(cv2.CAP_PROP_FPS)
                print(f"Set FPS to {fps}, actual: {actual_fps}")
            if mode is not None:
                possible = capture.set(cv2.CAP_PROP_MODE, mode)
                print('mode', mode, possible)

            # Set codec if specified
            if fourcc is not None:
                fcc = fourcc.upper()
                fourcc_code = cv2.VideoWriter_fourcc(*fcc)
                capture.set(cv2.CAP_PROP_FOURCC, fourcc_code)
                actual_fourcc = capture.get(cv2.CAP_PROP_FOURCC)
                actual_fourcc = int(actual_fourcc).to_bytes(4, "little").decode("ascii", errors="replace")
                print(f"cb_sensor_opencv2: Set fourcc to {fourcc}, actual: {actual_fourcc}")

            # Configure camera properties
            configure_camera_properties(capture, {
                'auto_exposure': auto_exposure,
                'exposure': exposure,
                'brightness': brightness,
                'contrast': contrast,
                'saturation': saturation,
                'hue': hue,
                'gain': gain
            }, verbose)

            ret, frame = capture.read()
            if not ret:
                print(f"cb_sensor_opencv2: Failed to read first frame from video source")
                print(f"capture.read() returned ret={ret}, frame={frame}")
                print(f"Capture state: isOpened={capture.isOpened()}")
                print(f"Last OpenCV error: {capture.get(cv2.CAP_PROP_POS_FRAMES)}")
                print(f"Buffer size: {capture.get(cv2.CAP_PROP_BUFFERSIZE)}")
                # Try to get more diagnostics
                try:
                    print(f"Backend name: {capture.getBackendName()}")
                    print(f"Camera API: {capture.get(cv2.CAP_PROP_API)}")
                except:
                    print("Could not get backend information")

                capture.release()
                capture = None
                return False

            actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = capture.get(cv2.CAP_PROP_FPS)
            backend_name = capture.getBackendName()
            print(f"  Source: {source}")
            print(f"  Backend: {backend_name}")
            print(f"  Resolution: {actual_width}x{actual_height}")
            print(f"  FPS: {actual_fps}")
            print(f"  Frame shape: {frame.shape}")
            print(f"  Frame dtype: {frame.dtype}")
            if frame.shape != states_color.shape:
                print(f"cb_sensor_opencv2: Frame shape {frame.shape} doesn't match state array shape {states_color.shape}")
            return True

        except Exception as e:
            print(f"cb_sensor_opencv2: Exception during video capture initialization: {e}")
            import traceback
            traceback.print_exc()
            if capture is not None:
                capture.release()
                capture = None
            return False

    def configure_camera_properties(capture, props: dict, verbose: bool = False):
        """Configure camera properties."""
        property_map = {
            'auto_exposure': (cv2.CAP_PROP_AUTO_EXPOSURE, 1 if props['auto_exposure'] else 0),
            'exposure': (cv2.CAP_PROP_EXPOSURE, props['exposure']),
            'brightness': (cv2.CAP_PROP_BRIGHTNESS, props['brightness']),
            'contrast': (cv2.CAP_PROP_CONTRAST, props['contrast']),
            'saturation': (cv2.CAP_PROP_SATURATION, props['saturation']),
            'hue': (cv2.CAP_PROP_HUE, props['hue']),
            'gain': (cv2.CAP_PROP_GAIN, props['gain']),
        }

        for prop_name, (prop_id, prop_value) in property_map.items():
            if prop_value is not None:
                try:
                    capture.set(prop_id, prop_value)
                    if verbose:
                        actual_value = capture.get(prop_id)
                        print(f"  {prop_name}: set to {prop_value}, actual: {actual_value}")
                except Exception as e:
                    if verbose:
                        print(f"  cb_sensor_opencv2: Failed to set {prop_name}: {e}")

    init_cap()

    def cb():
        nonlocal capture

        if capture is None:
            if not retry:
                print('cb_sensor_opencv2 exit no retry')
                return True
            result = init_cap()
            if verbose:
                print(f"cb_sensor_opencv2: Initialization result: {result}")

        ret, frame = capture.read()
        if not ret:
            print(f"cb_sensor_opencv2: Failed to capture frame from video source ret={ret}, frame={frame} isOpened={capture.isOpened()}")
            try:
                print(f"cb_sensor_opencv2: Current position: {capture.get(cv2.CAP_PROP_POS_FRAMES)}")
                print(f"Buffer size: {capture.get(cv2.CAP_PROP_BUFFERSIZE)}")
                print(f"Frame count: {capture.get(cv2.CAP_PROP_FRAME_COUNT)}")
                print(f"Backend: {capture.getBackendName()}")
            except:
                print("Could not get capture statistics")
            capture = None
            return

        if verbose:
            print(f"cb_sensor_opencv2: Successfully captured frame: shape={frame.shape}, dtype={frame.dtype}")

        # Convert frame based on image type
        processed_frame = frame
        # Ensure frame matches expected dimensions
        if processed_frame.shape != states_color.shape:
            # Try to resize frame to match state array
            try:
                processed_frame = cv2.resize(processed_frame, (states_color.shape[1], states_color.shape[0]))
            except Exception as e:
                print(f"cb_sensor_opencv2: Failed to resize frame: {e} Original shape: {frame.shape}, target shape: {states_color.shape}")
                return
        # bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_UYVY)
        states_color[:] = processed_frame

    return cb


if __name__ == "__main__":
    state_img = np.zeros((480, 640, 3), dtype=np.uint8)
    source = 4

    cb_sensor = cb_sensor_opencv2(
        states_color=state_img,
        source=source,
        resolution='640x480',
        verbose=False,
    )

    from unicon.io.opencv2 import cb_send_opencv2
    cb_disp = cb_send_opencv2(
        state_img=state_img,
    )

    for i in range(10000):
        cb_sensor()
        cb_disp()
            # time.sleep(0.1)
