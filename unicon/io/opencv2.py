import cv2


def cb_send_opencv2(
    window_name="default",
    window_layout='horizontal',  # 'horizontal', 'vertical', 'grid'
    window_size=None,
    wait_key=1,
    scale_factor=1.0,
    normalize_depth=True,
    depth_colormap=cv2.COLORMAP_JET,
    depth_range=None,  # (min_val, max_val) for depth normalization
    color_conversions=None,  # dict of cv2 color conversions per state key
    frame_processing=None,  # dict of custom processing functions per state key
    keys=None,
    **states,
):
    """
    Create a general OpenCV display callback for any state arrays (RealSense independent).
    This callback displays arbitrary state arrays using OpenCV, supporting different
    data types (depth, color, masks, etc.) and layouts.
    Args:
        window_name: Window title for OpenCV display
        window_layout: Layout arrangement ('horizontal', 'vertical', 'grid')
        window_size: Target window size (width, height) or None for auto
        wait_key: Wait time for cv2.waitKey() (0 for infinite wait, 1 for no wait)
        scale_factor: Scale factor for all frames (1.0 = original size)
        normalize_depth: Whether to normalize depth arrays for visualization
        depth_colormap: OpenCV colormap for depth visualization
        depth_range: Tuple (min_val, max_val) for depth normalization or None for auto
        color_conversions: Dict of color conversions per state key, e.g. {'color': cv2.COLOR_RGB2BGR}
        frame_processing: Dict of custom processing functions per state key
        **states: Keyword arguments containing state arrays to display
    Returns:
        Callback function that displays state arrays in OpenCV window
    Example:
        >>> import numpy as np
        >>> depth = np.random.rand(480, 640) * 1000  # Random depth data
        >>> color = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Random color
        >>> mask = np.random.randint(0, 2, (480, 640), dtype=np.uint8) * 255  # Binary mask
        >>>
        >>> cb = cb_opencv_display_general(
        ...     window_name="Multi-Sensor Display",
        ...     depth=depth,
        ...     color=color,
        ...     mask=mask,
        ...     color_conversions={'color': cv2.COLOR_RGB2BGR},
        ...     window_layout='grid'
        ... )
        >>> cb()  # Display the current state
    """
    import numpy as np
    if cv2 is None:
        raise RuntimeError('opencv-python required for display')
    # Create OpenCV window
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    if window_size is not None:
        cv2.resizeWindow(window_name, *window_size)
    # Setup layout parameters
    state_keys = list(states.keys()) if keys is None else keys
    num_states = len(state_keys)
    if num_states == 0:
        print("Warning: No state arrays provided to display callback")
        return lambda: True
    def process_frame(key, frame):
        """Process a single frame based on its type and key."""
        if frame is None:
            return None
        processed = frame.copy()
        # Apply custom processing if provided
        if frame_processing and key in frame_processing:
            try:
                processed = frame_processing[key](processed)
            except Exception as e:
                print(f"Error in custom processing for {key}: {e}")
        # Handle different data types
        if len(processed.shape) == 2:  # 2D array (depth, mask, grayscale)
            # Debug output for frame processing
            key_lower = key.lower()
            is_ir_like = 'ir' in key_lower
            # IR frames should never be treated as depth-like, even if normalize_depth is True
            is_depth_like = (normalize_depth and not is_ir_like) or ('depth' in key_lower)

            if is_depth_like:
                # Normalize depth-like data
                if depth_range:
                    min_val, max_val = depth_range
                else:
                    min_val, max_val = processed.min(), processed.max()
                if max_val > min_val:
                    processed = ((processed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    processed = np.zeros_like(processed, dtype=np.uint8)
                # Apply colormap
                processed = cv2.applyColorMap(processed, depth_colormap)
            else:
                # Grayscale - convert to 3-channel for display
                # Handle IR frames and other grayscale data
                if is_ir_like:
                    # IR frames are typically 8-bit grayscale, ensure proper display
                    if processed.dtype != np.uint8:
                        processed = ((processed - processed.min()) / (processed.max() - processed.min()) * 255).astype(np.uint8)
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        elif len(processed.shape) == 3:  # 3D array (color)
            # Apply color conversion if specified
            if color_conversions and key in color_conversions:
                try:
                    processed = cv2.cvtColor(processed, color_conversions[key])
                except Exception as e:
                    print(f"Error applying color conversion for {key}: {e}")
        # Apply scaling
        if scale_factor != 1.0:
            width = int(processed.shape[1] * scale_factor)
            height = int(processed.shape[0] * scale_factor)
            processed = cv2.resize(processed, (width, height))
        return processed
    
    
    def arrange_frames(frames):
        """Arrange frames according to layout."""
        valid_frames = [f for f in frames if f is not None]
        if not valid_frames:
            return None
        if len(valid_frames) == 1:
            return valid_frames[0]
        if window_layout == 'horizontal':
            # Arrange horizontally - resize to match height
            target_height = valid_frames[0].shape[0]
            resized_frames = []
            for frame in valid_frames:
                if frame.shape[0] != target_height:
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    new_width = int(target_height * aspect_ratio)
                    frame = cv2.resize(frame, (new_width, target_height))
                resized_frames.append(frame)
            return np.hstack(resized_frames)
        elif window_layout == 'vertical':
            # Arrange vertically - resize to match width
            target_width = valid_frames[0].shape[1]
            resized_frames = []
            for frame in valid_frames:
                if frame.shape[1] != target_width:
                    aspect_ratio = frame.shape[0] / frame.shape[1]
                    new_height = int(target_width * aspect_ratio)
                    frame = cv2.resize(frame, (new_height, target_width))
                resized_frames.append(frame)
            return np.vstack(resized_frames)
        elif window_layout == 'grid':
            # Arrange in grid layout
            cols = int(np.ceil(np.sqrt(len(valid_frames))))
            rows = int(np.ceil(len(valid_frames) / cols))
            # Find target dimensions for all frames
            target_height = max(f.shape[0] for f in valid_frames)
            target_width = max(f.shape[1] for f in valid_frames)
            # Resize all frames to target dimensions
            resized_frames = []
            for frame in valid_frames:
                if frame.shape[0] != target_height or frame.shape[1] != target_width:
                    frame = cv2.resize(frame, (target_width, target_height))
                resized_frames.append(frame)
            # Create grid rows
            grid_rows = []
            for i in range(rows):
                start_idx = i * cols
                end_idx = min(start_idx + cols, len(resized_frames))
                if start_idx < len(resized_frames):
                    row_frames = resized_frames[start_idx:end_idx]
                    row = np.hstack(row_frames)
                    grid_rows.append(row)
            # If we have incomplete rows, pad them to match row width
            if grid_rows:
                max_row_width = max(row.shape[1] for row in grid_rows)
                padded_rows = []
                for row in grid_rows:
                    if row.shape[1] < max_row_width:
                        # Pad with black pixels
                        padding = max_row_width - row.shape[1]
                        black_pad = np.zeros((row.shape[0], padding, 3), dtype=row.dtype)
                        row = np.hstack([row, black_pad])
                    padded_rows.append(row)
                return np.vstack(padded_rows)
            return np.vstack(grid_rows) if grid_rows else None
        else:
            # Default to horizontal
            return np.hstack(valid_frames)
    
    
    def cb():
        try:
            processed_frames = []
            for key in state_keys:
                frame = states[key]
                processed = process_frame(key, frame)
                processed_frames.append(processed)
            display_frame = arrange_frames(processed_frames)
            if display_frame is not None:
                cv2.imshow(window_name, display_frame)
                # Handle key press
                key = cv2.waitKey(wait_key) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    return True
        except Exception as e:
            print(f"Display error: {e}")
            # return True
    return cb
