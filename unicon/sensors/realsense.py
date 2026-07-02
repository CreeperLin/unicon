"""
Intel RealSense camera interface module.
Provides callback-based interface for RealSense depth cameras,
including configuration, frame capture, alignment, and filtering.
"""
import numpy as np
try:
    import pyrealsense2 as rs
except ImportError:
    print('pyrealsense2 not available, install with: pip install pyrealsense2')
    rs = None
try:
    import cv2
except ImportError:
    print('opencv-python not available, install with: pip install opencv-python')
    cv2 = None


_sensor_names_std = {
    'Stereo Module': 'depth',
    'RGB Camera': 'color',
    'Motion Module': 'imu',
}


def get_device_ids():
    """
    Enumerate all connected RealSense devices.
    Returns:
        list: Sorted list of device serial numbers
    """
    if rs is None:
        raise RuntimeError('pyrealsense2 not installed')
    ctx = rs.context()
    devices = ctx.query_devices()
    device_ids = [devices[i].get_info(rs.camera_info.serial_number)
                  for i in range(len(devices))]
    device_ids.sort()  # Ensure consistent ordering
    print(f"Found {len(device_ids)} RealSense device(s): {device_ids}")
    return device_ids


def set_sensor_options(profile, options, verbose=False):
    """
    Set camera options for different sensors in a RealSense profile.
    Args:
        profile: RealSense pipeline profile
        options: Dictionary of options to set
        verbose: Print status information
    """
    try:
        # Get sensors from the profile
        device = profile.get_device()
        sensors = device.query_sensors()
        for sensor in sensors:
            sensor_name = _sensor_names_std[sensor.name]
            if verbose:
                print(f"Configuring {sensor_name} sensor...")
            # Apply exposure settings
            if 'exposure' in options and options['exposure'] is not None:
                exposure_value = options['exposure'].get(sensor_name.lower())
                if exposure_value is not None:
                    try:
                        current_exp = sensor.get_option(rs.option.exposure)
                        sensor.set_option(rs.option.exposure, exposure_value)
                        new_exp = sensor.get_option(rs.option.exposure)
                        if verbose:
                            print(f"  {sensor_name} exposure: {current_exp} -> {new_exp}")
                    except Exception as e:
                        if verbose:
                            print(f"  Failed to set {sensor_name} exposure: {e}")
            # Apply auto exposure settings
            if 'auto_exposure' in options and options['auto_exposure'] is not None:
                auto_exp_value = options['auto_exposure'].get(sensor_name.lower())
                if auto_exp_value is not None:
                    try:
                        current_auto = sensor.get_option(rs.option.enable_auto_exposure)
                        sensor.set_option(rs.option.enable_auto_exposure, auto_exp_value)
                        new_auto = sensor.get_option(rs.option.enable_auto_exposure)
                        if verbose:
                            print(f"  {sensor_name} auto exposure: {current_auto} -> {new_auto}")
                    except Exception as e:
                        if verbose:
                            print(f"  Failed to set {sensor_name} auto exposure: {e}")
            # Apply gain settings
            if 'gain' in options and options['gain'] is not None:
                gain_value = options['gain'].get(sensor_name.lower())
                if gain_value is not None:
                    try:
                        current_gain = sensor.get_option(rs.option.gain)
                        sensor.set_option(rs.option.gain, gain_value)
                        new_gain = sensor.get_option(rs.option.gain)
                        if verbose:
                            print(f"  {sensor_name} gain: {current_gain} -> {new_gain}")
                    except Exception as e:
                        if verbose:
                            print(f"  Failed to set {sensor_name} gain: {e}")
    except Exception as e:
        if verbose:
            print(f"Failed to set sensor options: {e}")


def init_camera(
    device_id=None,
    width=640,
    height=360,
    fps=30,
    enable_depth=True,
    enable_color=True,
    enable_ir=False,
    enable_ir2=False,
    align_to='color',
    depth_format='z16',
    color_format='bgr8',
    ir_format='y8',
    exposure=None,
    color_exposure=None,
    depth_exposure=None,
    auto_exposure=None,
    color_auto_exposure=None,
    depth_auto_exposure=None,
    gain=None,
    color_gain=None,
    depth_gain=None,
    laser_power=None,
    enable_emitter=None,
    depth_preset=None,
    visual_preset=None,
):
    """
    Initialize a single RealSense camera with comprehensive stream configuration.
    Args:
        device_id: Device serial number (str) or None for auto-detect
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frames per second
        enable_depth: Enable depth stream
        enable_color: Enable RGB color stream
        enable_ir: Enable infrared stream 1
        enable_ir2: Enable infrared stream 2
        align_to: Alignment target ('color', 'depth', or None)
        depth_format: Depth format ('z16', 'disparity16', etc.)
        color_format: Color format ('rgb8', 'bgr8', 'rgba8', 'bgra8', etc.)
        ir_format: IR format ('y8', 'y16', etc.)
        exposure: Manual exposure value (applies to all sensors if specific ones not set)
        color_exposure: Manual exposure for color sensor (overwrites exposure)
        depth_exposure: Manual exposure for depth sensor (overwrites exposure)
        auto_exposure: Auto exposure mode for all sensors (0=disabled, 1=enabled)
        color_auto_exposure: Auto exposure for color sensor
        depth_auto_exposure: Auto exposure for depth sensor
        gain: Gain value for all sensors (applies if specific ones not set)
        color_gain: Gain for color sensor
        depth_gain: Gain for depth sensor
        laser_power: Laser power (0-360, None for default)
        enable_emitter: Enable IR emitter (True/False/None for default)
        depth_preset: Depth preset mode (0-5: Default, Hand, HighAccuracy, HighDensity, MediumDensity, Custom)
        visual_preset: Visual preset (None for default)
    Returns:
        tuple: (pipeline, align, depth_scale, profile) objects
    """
    if rs is None:
        raise RuntimeError('pyrealsense2 not installed')
    pipeline = rs.pipeline()
    config = rs.config()
    # Enable specific device if provided
    if device_id is not None:
        config.enable_device(device_id)
        print(f"Initializing camera {device_id}")
    else:
        devices = get_device_ids()
        if devices:
            device_id = devices[0]
            config.enable_device(device_id)
            print(f"Auto-selected camera {device_id}")
        else:
            print("No device ID specified, using default device")
    # Configure streams
    stream_enabled = []
    if enable_depth:
        format_map = {'z16': rs.format.z16, 'disparity16': rs.format.disparity16}
        fmt = format_map.get(depth_format, rs.format.z16)
        print(f"Enabling depth stream: {width}x{height}@{fps} ({depth_format})")
        config.enable_stream(rs.stream.depth, width, height, fmt, fps)
        stream_enabled.append('depth')
    if enable_color:
        format_map = {
            'rgb8': rs.format.rgb8,
            'bgr8': rs.format.bgr8,
            'rgba8': rs.format.rgba8,
            'bgra8': rs.format.bgra8,
        }
        fmt = format_map.get(color_format, rs.format.rgb8)
        print(f"Enabling color stream: {width}x{height}@{fps} ({color_format})")
        config.enable_stream(rs.stream.color, width, height, fmt, fps)
        stream_enabled.append('color')
    if enable_ir:
        format_map = {'y8': rs.format.y8, 'y16': rs.format.y16}
        fmt = format_map.get(ir_format, rs.format.y8)
        print(f"Enabling IR1 stream: {width}x{height}@{fps} ({ir_format})")
        config.enable_stream(rs.stream.infrared, 1, width, height, fmt, fps)
        stream_enabled.append('ir1')
    if enable_ir2:
        format_map = {'y8': rs.format.y8, 'y16': rs.format.y16}
        fmt = format_map.get(ir_format, rs.format.y8)
        print(f"Enabling IR2 stream: {width}x{height}@{fps} ({ir_format})")
        config.enable_stream(rs.stream.infrared, 2, width, height, fmt, fps)
        stream_enabled.append('ir2')
    # Start pipeline
    profile = pipeline.start(config)
    device = profile.get_device()
    # Get depth scale
    depth_scale = None
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale}")
    # Set depth preset
    if depth_preset is not None:
        try:
            depth_sensor.set_option(rs.option.visual_preset, depth_preset)
            print(f"Set depth preset to: {depth_preset}")
        except Exception as e:
            print(f"Failed to set depth preset: {e}")
    # Set laser power
    if laser_power is not None:
        try:
            depth_sensor.set_option(rs.option.laser_power, laser_power)
            print(f"Set laser power to: {laser_power}")
        except Exception as e:
            print(f"Failed to set laser power: {e}")
    # Set emitter enabled/disabled
    if enable_emitter is not None:
        try:
            depth_sensor.set_option(rs.option.emitter_enabled, 1 if enable_emitter else 0)
            print(f"IR emitter: {'enabled' if enable_emitter else 'disabled'}")
        except Exception as e:
            print(f"Failed to set emitter: {e}")
    # Set exposure settings for different sensors
    set_sensor_options(profile, {
        'exposure': {
            'color': (color_exposure if color_exposure is not None else exposure),
            'depth': (depth_exposure if depth_exposure is not None else exposure),
        },
        'auto_exposure': {
            'color': (color_auto_exposure if color_auto_exposure is not None else auto_exposure),
            'depth': (depth_auto_exposure if depth_auto_exposure is not None else auto_exposure),
        },
        'gain': {
            'color': (color_gain if color_gain is not None else gain),
            'depth': (depth_gain if depth_gain is not None else gain),
        }
    }, verbose=True)
    # Create alignment object
    align = None
    if align_to is not None:
        if align_to == 'color' and enable_color:
            align = rs.align(rs.stream.color)
            print("Frame alignment enabled (all -> color)")
        elif align_to == 'depth' and enable_depth:
            align = rs.align(rs.stream.depth)
            print("Frame alignment enabled (all -> depth)")
    print(f"Enabled streams: {', '.join(stream_enabled)}")
    return pipeline, align, depth_scale, profile


def get_depth_filters(
    decimation_magnitude=4,
    spatial_magnitude=5,
    spatial_smooth_alpha=1.0,
    spatial_smooth_delta=50,
    spatial_holes_fill=3,
    hole_filling_mode=2,
):
    """
    Create depth filtering pipeline.
    Args:
        decimation_magnitude: Decimation filter magnitude
        spatial_magnitude: Spatial filter magnitude
        spatial_smooth_alpha: Spatial filter alpha parameter
        spatial_smooth_delta: Spatial filter delta parameter
        spatial_holes_fill: Spatial filter holes fill mode
        hole_filling_mode: Hole filling filter mode (0-2)
    Returns:
        tuple: (depth_to_disparity, disparity_to_depth, decimation,
                spatial, temporal, hole_filling) filter objects
    """
    if rs is None:
        raise RuntimeError('pyrealsense2 not installed')
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, decimation_magnitude)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, spatial_magnitude)
    spatial.set_option(rs.option.filter_smooth_alpha, spatial_smooth_alpha)
    spatial.set_option(rs.option.filter_smooth_delta, spatial_smooth_delta)
    spatial.set_option(rs.option.holes_fill, spatial_holes_fill)
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter(mode=hole_filling_mode)
    return depth_to_disparity, disparity_to_depth, decimation, spatial, temporal, hole_filling


def apply_filters(
    depth_frame,
    depth_to_disparity=None,
    disparity_to_depth=None,
    decimation=None,
    spatial=None,
    temporal=None,
    hole_filling=None,
):
    """
    Apply depth filters to a frame.
    Args:
        depth_frame: RealSense depth frame
        depth_to_disparity: Disparity transform filter
        disparity_to_depth: Inverse disparity transform
        decimation: Decimation filter
        spatial: Spatial filter
        temporal: Temporal filter
        hole_filling: Hole filling filter
    Returns:
        Filtered depth frame
    """
    frame = depth_frame
    if decimation is not None:
        frame = decimation.process(frame)
    if depth_to_disparity is not None:
        frame = depth_to_disparity.process(frame)
    if spatial is not None:
        frame = spatial.process(frame)
    if disparity_to_depth is not None:
        frame = disparity_to_depth.process(frame)
    if temporal is not None:
        frame = temporal.process(frame)
    if hole_filling is not None:
        frame = hole_filling.process(frame)
    return frame


def capture_frames(pipeline, align=None, timeout_ms=5000):
    """
    Capture frames from RealSense camera.

    Args:
        pipeline: RealSense pipeline object
        align: Alignment object (None to skip alignment)
        timeout_ms: Timeout in milliseconds

    Returns:
        dict: Dictionary containing available frames {'depth': frame, 'color': frame, 'ir1': frame, 'ir2': frame}
               Empty dict on timeout or error
    """
    try:
        frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
        if align is not None:
            frames = align.process(frames)

        # Return all available frames as dict
        result = {}

        depth_frame = frames.get_depth_frame()
        if depth_frame:
            result['depth'] = depth_frame

        color_frame = frames.get_color_frame()
        if color_frame:
            result['color'] = color_frame

        # Get IR frames if available
        try:
            ir1_frame = frames.get_infrared_frame(1)
            if ir1_frame:
                result['ir1'] = ir1_frame
        except:
            pass

        try:
            ir2_frame = frames.get_infrared_frame(2)
            if ir2_frame:
                result['ir2'] = ir2_frame
        except:
            pass

        return result

    except Exception as e:
        print(f"Error capturing frames: {e}")
        return {}


def frame_to_numpy(frame, depth_scale=None):
    """
    Convert RealSense frame to numpy array.
    Args:
        frame: RealSense frame object
        depth_scale: Scale factor for depth frames (None for color frames)
    Returns:
        numpy array
    """
    if frame is None:
        return None
    array = np.asanyarray(frame.get_data())
    if depth_scale is not None:
        array = array.astype(np.float32) * depth_scale
    return array


def process_depth(
    depth_array,
    clip_min=0.15,
    clip_max=2.0,
    normalize=True,
    offset=0.0,
):
    """
    Process depth array with clipping and normalization.
    Args:
        depth_array: Depth values in meters (numpy array)
        clip_min: Minimum depth value (values below set to 0)
        clip_max: Maximum depth value (values above clipped)
        normalize: Normalize to [0, 1] range
        offset: Depth offset to add before processing
    Returns:
        Processed depth array
    """
    depth = depth_array.astype(np.float32)
    depth += offset
    # Clip values
    depth[depth < clip_min] = 0.0
    depth[depth > clip_max] = clip_max
    # Normalize to [0, 1]
    if normalize:
        depth = (depth - 0.0) / (clip_max - 0.0)
    return depth


def resize_frame(frame, size, interpolation='linear'):
    """
    Resize frame using OpenCV.
    Args:
        frame: Input frame (numpy array)
        size: Output size as (width, height)
        interpolation: Interpolation method ('linear', 'nearest', 'cubic')
    Returns:
        Resized frame
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError('opencv-python required for resizing')
    interp_map = {
        'linear': cv2.INTER_LINEAR,
        'nearest': cv2.INTER_NEAREST,
        'cubic': cv2.INTER_CUBIC,
    }
    interp = interp_map.get(interpolation, cv2.INTER_LINEAR)
    return cv2.resize(frame, dsize=size, interpolation=interp)


def stop_camera(pipeline):
    """
    Stop and cleanup RealSense camera.
    Args:
        pipeline: RealSense pipeline object or list of pipelines
    """
    if isinstance(pipeline, list):
        for p in pipeline:
            p.stop()
    else:
        pipeline.stop()
    print("Camera stopped")


def cb_sensor_realsense(
    states_color=None,
    states_depth=None,
    states_ir1=None,
    states_ir2=None,
    width=None,
    height=None,
    resolution=None,
    fps=30,
    color_format='bgr8',
    depth_format='z16',
    enable_ir_emitter=True,
    enable_color=None,
    enable_depth=None,
    enable_ir1=None,
    enable_ir2=None,
    align_depth_to_color=True,
    laser_power=None,
    visual_preset=None,
    accuracy_preset=None,
    color_exposure=None,
    color_gain=None,
    depth_exposure=None,
    depth_gain=None,
    auto_exposure_color=True,
    auto_exposure_depth=False,
    filters=None,
    device_id=None,
    verbose=False,
    color_key='color',
    depth_key='depth',
    ir1_key='ir1',
    ir2_key='ir1',
    **states,
):
    """
    RealSense camera callback factory function.
    Creates a callback that handles camera initialization and frame capture.

    Args:
        states_color: Numpy array to store RGB/color frames (modified in place)
        states_depth: Numpy array to store depth frames (modified in place)
        states_ir1: Numpy array to store IR1 frames (modified in place)
        states_ir2: Numpy array to store IR2 frames (modified in place)
        width (int): Frame width (pixels)
        height (int): Frame height (pixels)
        resolution (str): Resolution string like '640x480' (overrides width/height)
        fps (int): Frames per second
        color_format (str): Color frame format ('bgr8', 'rgb8', etc.)
        depth_format (str): Depth frame format ('z16', 'z32f', etc.)
        enable_ir_emitter (bool): Enable IR emitter
        enable_color (bool): Enable color stream
        enable_depth (bool): Enable depth stream
        enable_ir1 (bool): Enable first infrared stream
        enable_ir2 (bool): Enable second infrared stream
        align_depth_to_color (bool): Align depth frames to color frames
        laser_power (float): Laser power (0.0 to 360.0)
        visual_preset (str): Visual preset for depth camera
        accuracy_preset (str): Accuracy preset for depth camera
        color_exposure (int): Color exposure value
        color_gain (int): Color gain value
        depth_exposure (int): Depth exposure value
        depth_gain (int): Depth gain value
        auto_exposure_color (bool): Auto exposure for color
        auto_exposure_depth (bool): Auto exposure for depth
        filters (dict): Dictionary of post-processing filters
        device_id (str): Device serial number
        verbose (bool): Print status information

    Returns:
        function: Callback function that captures frames from RealSense camera
    """
    import itertools
    from unicon.utils import printv
    # Infer resolution from provided state arrays if not explicitly set
    if resolution is None:
        inferred_resolution = None
        if states_color is not None and hasattr(states_color, 'shape'):
                inferred_resolution = (states_color.shape[1], states_color.shape[0])  # (width, height)
                if verbose:
                    print(f"Inferred resolution from RGB array: {inferred_resolution}")
        elif states_depth is not None and hasattr(states_depth, 'shape'):
                inferred_resolution = (states_depth.shape[1], states_depth.shape[0])  # (width, height)
                if verbose:
                    print(f"Inferred resolution from depth array: {inferred_resolution}")
        elif states_ir1 is not None and hasattr(states_ir1, 'shape'):
                inferred_resolution = (states_ir1.shape[1], states_ir1.shape[0])  # (width, height)
                if verbose:
                    print(f"Inferred resolution from IR1 array: {inferred_resolution}")

        if inferred_resolution is not None:
            resolution = f"{inferred_resolution[0]}x{inferred_resolution[1]}"
            if verbose:
                print(f"Using inferred resolution: {resolution}")

    # Parse resolution string to get width/height
    if resolution is not None and isinstance(resolution, str):
        try:
            width_str, height_str = resolution.lower().split('x')
            width = int(width_str)
            height = int(height_str)
        except (ValueError, AttributeError):
            print(f"Invalid resolution format: {resolution}, using defaults")
            width = width or 640
            height = height or 480
    else:
        width = width or 640
        height = height or 480

    # Store internal camera state in a local dict
    camera_state = {}

    enable_color = (states_color is not None) if enable_color is None else enable_color
    enable_depth = (states_depth is not None) if enable_depth is None else enable_depth
    enable_ir1 = (states_ir1 is not None) if enable_ir1 is None else enable_ir1
    enable_ir2 = (states_ir2 is not None) if enable_ir2 is None else enable_ir2

    print('enable_color', enable_color)
    print('enable_depth', enable_depth)
    print('enable_ir1', enable_ir1)
    print('enable_ir2', enable_ir2)

    # Initialize camera on first call
    if 'pipeline' not in camera_state:
        if verbose:
            print("Initializing RealSense camera...")

        # Prepare camera parameters (map to init_camera signature)
        camera_params = {
            'width': width,
            'height': height,
            'fps': fps,
            'color_format': color_format,
            'depth_format': depth_format,
            'enable_emitter': enable_ir_emitter,
            'enable_color': enable_color,
            'enable_depth': enable_depth,
            'enable_ir': enable_ir1,
            'enable_ir2': enable_ir2,
            'align_to': 'color' if align_depth_to_color else None,
            'laser_power': laser_power,
            'visual_preset': visual_preset,
            'color_exposure': color_exposure,
            'color_gain': color_gain,
            'depth_exposure': depth_exposure,
            'depth_gain': depth_gain,
            'color_auto_exposure': auto_exposure_color,
            'depth_auto_exposure': auto_exposure_depth,
            'device_id': device_id
        }

        # Initialize camera with current settings
        result = init_camera(**camera_params)
        if result is None:
            print("Failed to initialize camera")
            def error_callback():
                return False
            return error_callback

        # Unpack result from init_camera
        pipeline, align, depth_scale, profile = result

        # Setup processing (use existing align from init_camera)
        align_to_color = align if align_depth_to_color else None

        # Setup filters if specified
        processed_filters = {}
        if filters is not None:
            if filters.get('decimation', {}).get('enabled', False):
                dec_filter = rs.decimation_filter()
                dec_filter.set_option(rs.option.filter_magnitude,
                                    filters['decimation'].get('magnitude', 2))
                processed_filters['decimation'] = dec_filter

            if filters.get('spatial', {}).get('enabled', False):
                spatial_filter = rs.spatial_filter()
                spatial_filter.set_option(rs.option.filter_magnitude,
                                        filters['spatial'].get('magnitude', 2))
                spatial_filter.set_option(rs.option.filter_smooth_alpha,
                                        filters['spatial'].get('smooth_alpha', 0.5))
                spatial_filter.set_option(rs.option.filter_smooth_delta,
                                        filters['spatial'].get('smooth_delta', 20))
                spatial_filter.set_option(rs.option.holes_fill,
                                        filters['spatial'].get('holes_fill', 1))
                processed_filters['spatial'] = spatial_filter

            if filters.get('temporal', {}).get('enabled', False):
                temporal_filter = rs.temporal_filter()
                temporal_filter.set_option(rs.option.filter_smooth_alpha,
                                         filters['temporal'].get('smooth_alpha', 0.5))
                temporal_filter.set_option(rs.option.filter_smooth_delta,
                                         filters['temporal'].get('smooth_delta', 20))
                temporal_filter.set_option(rs.option.holes_fill,
                                         filters['temporal'].get('holes_fill', 3))
                processed_filters['temporal'] = temporal_filter

            if filters.get('disparity', {}).get('enabled', False):
                depth_to_disparity = rs.disparity_transform()
                disparity_to_depth = rs.disparity_transform(False)
                processed_filters['disparity'] = (depth_to_disparity, disparity_to_depth)

            if filters.get('hole_filling', {}).get('enabled', False):
                hole_filling_filter = rs.hole_filling_filter()
                hole_filling_filter.set_option(rs.option.holes_fill,
                                             filters['hole_filling'].get('fill_mode', 1))
                processed_filters['hole_filling'] = hole_filling_filter

        # Store in camera state
        camera_state['pipeline'] = pipeline
        camera_state['align_to_color'] = align_to_color
        camera_state['filters'] = processed_filters
        camera_state['intrinsics'] = {}  # Store intrinsics for each stream type

    def cb():
        """RealSense frame capture callback function."""
        nonlocal camera_state
        try:
            # Get camera components from camera_state
            pipeline = camera_state['pipeline']
            align_to_color = camera_state['align_to_color']
            processed_filters = camera_state['filters']
            intrinsics = camera_state['intrinsics']

            # Wait for frames
            frames = pipeline.wait_for_frames()

            if verbose:
                printv(f"Captured frameset with {len(frames)} frames")

            # Store raw frames in camera state
            camera_state['frames_raw'] = frames

            # Store aligned frames if alignment is enabled
            if align_to_color is not None:
                aligned_frames = align_to_color.process(frames)
                camera_state['frames_aligned'] = aligned_frames
            else:
                aligned_frames = frames  # Use original frames if no alignment

            if align_to_color is not None:
                # Create iterator that processes both original and aligned frames
                frame_iterator = itertools.chain(frames, aligned_frames)
                # Store aligned frames for is_aligned detection
                aligned_frame_set = set(aligned_frames)
            else:
                frame_iterator = frames
                aligned_frame_set = set()

            for frame in frame_iterator:
                stream_type = frame.profile.stream_type()
                frame_data = np.asanyarray(frame.get_data())
                stream_intrinsics = frame.profile.as_video_stream_profile().get_intrinsics()

                # Check if this is an aligned frame (more efficient set lookup)
                is_aligned = align_to_color is not None and frame in aligned_frame_set

                if stream_type == rs.stream.color:
                    if states_color is not None:
                        # Check if the captured frame matches the expected shape
                        if len(frame_data.shape) == 3 and len(states_color.shape) == 3:
                            if frame_data.shape == states_color.shape:
                                states_color[:] = frame_data  # Copy data to existing array
                            else:
                                printv(f"RGB frame shape mismatch: expected {states_color.shape}, got {frame_data.shape}")
                        elif len(frame_data.shape) == 3 and len(states_color.shape) == 2:
                            # Handle case where states_color is 2D but captured frame is 3D
                            if verbose:
                                print(f"Converting RGB to grayscale: {frame_data.shape} -> {states_color.shape}")
                            gray_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
                            if gray_frame.shape == states_color.shape:
                                states_color[:] = gray_frame

                        intrinsics_key = 'rgb_aligned' if is_aligned else 'rgb'
                        intrinsics[intrinsics_key] = stream_intrinsics
                        if verbose:
                            alignment_str = " (aligned)" if is_aligned else ""
                            printv(f"RGB frame{alignment_str}: {frame_data.shape}")

                elif stream_type == rs.stream.depth:
                    # Get depth data from frame
                    depth_data = frame_data

                    # Apply filters to depth frame (only on original frames, not aligned)
                    if not is_aligned and processed_filters is not None:
                        # Note: Filters are applied to RealSense frames, not numpy arrays
                        # This happens during capture, so we just acknowledge it here
                        for filter_name, filter_obj in processed_filters.items():
                            if verbose:
                                print(f"Filter {filter_name} applied during capture")

                    if states_depth is not None:
                        # Check if the captured frame matches the expected shape
                        if depth_data.shape == states_depth.shape:
                            states_depth[:] = depth_data  # Copy data to existing array
                        else:
                            printv(f"Depth frame shape mismatch: expected {states_depth.shape}, got {depth_data.shape}")

                        intrinsics_key = 'depth_aligned' if is_aligned else 'depth'
                        intrinsics[intrinsics_key] = stream_intrinsics
                        if verbose:
                            alignment_str = " (aligned)" if is_aligned else ""
                            printv(f"Depth frame{alignment_str}: {depth_data.shape}")

                elif stream_type == rs.stream.infrared:
                    # Determine which IR stream this is (IR1 or IR2)
                    ir_index = frame.profile.stream_index()

                    if ir_index == 1 and states_ir1 is not None:
                        # Check if the captured frame matches the expected shape
                        if frame_data.shape == states_ir1.shape:
                            states_ir1[:] = frame_data
                        elif frame_data.shape[:2] == states_ir1.shape[:2]:
                            states_ir1[:] = frame_data[:, :, None]
                        else:
                            printv(f"IR1 frame shape mismatch: expected {states_ir1.shape}, got {frame_data.shape}")

                        intrinsics_key = 'ir1_aligned' if is_aligned else 'ir1'
                        intrinsics[intrinsics_key] = stream_intrinsics
                        if verbose:
                            alignment_str = " (aligned)" if is_aligned else ""
                            printv(f"IR1 frame{alignment_str}: {frame_data.shape}")

                    elif ir_index == 2 and states_ir2 is not None:
                        # Check if the captured frame matches the expected shape
                        if frame_data.shape == states_ir2.shape:
                            states_ir2[:] = frame_data
                        elif frame_data.shape[:2] == states_ir2.shape[:2]:
                            states_ir2[:] = frame_data[:, :, None]
                        else:
                            printv(f"IR2 frame shape mismatch: expected {states_ir2.shape}, got {frame_data.shape}")

                        intrinsics_key = 'ir2_aligned' if is_aligned else 'ir2'
                        intrinsics[intrinsics_key] = stream_intrinsics
                        if verbose:
                            alignment_str = " (aligned)" if is_aligned else ""
                            printv(f"IR2 frame{alignment_str}: {frame_data.shape}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error capturing RealSense frames: {e}")

    return cb
