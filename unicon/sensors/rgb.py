import os
import argparse
import time

import numpy as np
import pyrealsense2 as rs
import cv2
from scipy.spatial.transform import Rotation as R

from unicon.states import states_get, states_init

CIRCLE_RADIUS = 1.0         # 圆周半径（米）
Z_CLIP = 0.2                # z轴clip范围（米）

def open_camera(infrared, width=640, height=480, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    if infrared:
        config.enable_stream(rs.stream.infrared, width, height, rs.format.y8, fps)
    else:
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    profile = pipeline.start(config)
    if infrared:
        infrared_profile = profile.get_stream(rs.stream.infrared)
        intrinsics = infrared_profile.as_video_stream_profile().get_intrinsics()
    else:
        color_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

    camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                              [0, intrinsics.fy, intrinsics.ppy],
                              [0, 0, 1]])
    dist_coeffs = np.array(intrinsics.coeffs)

    return pipeline, camera_matrix, dist_coeffs

def create_detector():
    # Define the dictionary and parameters for AprilTag detection
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    return detector

def draw_pose(frame, corners, rvec, tvec, camera_matrix, dist_coeffs, tag_size, tag_id, tag_pose=None):
    # draw corners
    cv2.polylines(frame, [corners.astype(int)], True, (0, 255, 0), 2)
    
    # draw tag axes
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, tag_size/2)
    
    # draw tag id
    center = corners.mean(axis=0).astype(int)
    cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)
    cv2.putText(frame, f"ID:{tag_id}", (center[0], center[1] - 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # draw pos
    if tag_pose is not None:
        _trans = tag_pose[0:3]
        position_text = f"Pos:({_trans[0]:.2f},{_trans[1]:.2f},{_trans[2]:.2f})m"
        cv2.putText(frame, position_text, (center[0], center[1] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # draw distance
    distance = np.linalg.norm(tvec)
    distance_text = f"Dist:{distance:.2f}m"
    cv2.putText(frame, distance_text, (center[0], center[1] + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return frame

def calculate_target_from_tag(corners, rvec, tvec, T_upstream, T_downstream, circle_radius=CIRCLE_RADIUS, z_clip=Z_CLIP):
    rot_matrix, _ = cv2.Rodrigues(rvec[:, 0])

    T_tag_in_cam = np.eye(4)
    T_tag_in_cam[0:3, 0:3] = rot_matrix
    T_tag_in_cam[0:3, 3] = tvec[:, 0]

    T_target_in_robot = T_upstream @ T_tag_in_cam @ T_downstream

    pos = T_target_in_robot[0:3, 3]
    dist = np.linalg.norm(pos[:2])
    if dist > circle_radius:
        theta = np.arctan2(pos[1], pos[0])
        pos[:2] = circle_radius *  * np.array([np.cos(theta), np.sin(theta)])
    pos[2] = np.clip(pos[2], -z_clip, z_clip)
    rot = R.from_matrix(T_target_in_robot[0:3, 0:3])
    quat = rot.as_quat()

    target = np.concatenate((pos, quat))
    return target

def calculate_T_upstream():

    # T_camlink_in_robot = np.array([
    #     [ 0.50475008,  0.63909876,  0.58032761,  0.06081622],
    #     [ 0.16730993,  0.58707733, -0.79205277, -0.29668969],
    #     [-0.84689713,  0.49688327,  0.18939976,  0.35160208],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]
    # ])

    T_camlink_in_robot = np.array([[0.674302, 0.000000, 0.738455, 0.097214,],
    [-0.000000, 1.000000, -0.000000, 0.019770,],
    [-0.738455, -0.000000, 0.674302, 0.281950,],
    [0.000000, 0.000000, 0.000000, 1.000000,]])
    # T_camlink_in_robot = np.eye(4)

    Ry_pos = R.from_euler('y', 90, degrees=True).as_matrix()
    Rx_neg = R.from_euler('x', -90, degrees=True).as_matrix()

    T_cam_in_camlink = np.eye(4)
    T_cam_in_camlink[0:3, 0:3] = Rx_neg @ Ry_pos

    T_cam_in_robot = T_camlink_in_robot @ T_cam_in_camlink

    return T_cam_in_robot

def calculate_T_downstream():
    
    Rx_neg = R.from_euler('x', -90, degrees=True).as_matrix()
    Ry_pos = R.from_euler('y', 90, degrees=True).as_matrix()
    Rx_pos = R.from_euler('x', 90, degrees=True).as_matrix()
    Ry_neg = R.from_euler('y', -90, degrees=True).as_matrix()
    Rz_pos = R.from_euler('z', 90, degrees=True).as_matrix()

    T_target_in_tag_pos = np.eye(4)
    # T_target_in_tag_pos[3, 0:3] = np.array([[0, -0.032, 0]])
    T_target_in_tag_pos[3, 0:3] = np.array([[0, -0.0, 0]])
    
    T_target_in_tag_rot = np.eye(4)
    # T_target_in_tag_rot[0:3, 0:3] = Ry_neg @ Rx_pos
    # T_target_in_tag_rot[0:3, 0:3] = Rx_neg @ Ry_pos
    T_target_in_tag_rot[0:3, 0:3] = Rz_pos
    T_target_in_tag = T_target_in_tag_rot @ T_target_in_tag_pos

    return T_target_in_tag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ut', '--unit_test', action='store_true', help='unit_test mode')
    parser.add_argument('-fc', '--fake_camera', action='store_true', help='use a test image as camera')
    parser.add_argument('-sd', '--skip_detect', action='store_true', help='skip detect')
    parser.add_argument('-ltid', '--left_hand_tag_id', type=int, default=1, help='left hand tag id')
    parser.add_argument('-rtid', '--right_hand_tag_id', type=int, default=2, help='right hand tag id')
    parser.add_argument('-ts', '--tag_size', type=float, default=0.0456, help='tag size in meters')
    parser.add_argument('-ifr', '--infrared', action='store_true', help='use infrared stream')
    parser.add_argument('-hdls', '--headless', action='store_true', help='run in headless mode')
    args = parser.parse_args()

    headless = args.headless

    if args.fake_camera:
        test_image_path = os.path.join(os.getcwd(), 'sensors', 'markers_36h11_table_0.png')
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            print(f"Failed to load test image from {test_image_path}")
            return
        
        # default camera_matrix and dist_coeffs for test image
        camera_matrix = np.array([[600, 0, test_image.shape[1]/2],
                                  [0, 600, test_image.shape[0]/2],
                                  [0, 0, 1]])
        dist_coeffs = np.zeros(5)
    else:
        pipeline, camera_matrix, dist_coeffs = open_camera(args.infrared)
    print("camera matrix:", camera_matrix)
    print("dist_coeffs:", dist_coeffs)

    detector = create_detector()
    T_cam_in_robot = calculate_T_upstream()
    T_target_in_tag = calculate_T_downstream()

    if not args.unit_test:
        states_init(use_shm=True, load=True, reuse=True)
    else:
        print("Unit test mode: states not initialized")


    # Initialize reference points of marker
    tag_size = args.tag_size
    half_size = tag_size / 2.0
    # By default, coordinate system origin is placed at the bottom-left corner, we change it to the center of the tag, see https://stackoverflow.com/questions/77559301/aruco-code-pose-estimation-solvepnp-issues
    obj_points = np.array([
        [-half_size, half_size, 0],   # upper left
        [half_size, half_size, 0],    # upper right
        [half_size, -half_size, 0],   # lower right
        [-half_size, -half_size, 0]   # lower left
    ], dtype=np.float32)
    
    try:
        while True:
            start_time = time.time()

            # acquire frame
            if args.fake_camera:
                color_image = test_image.copy()
            else:
                frames = pipeline.wait_for_frames()
                if args.infrared:
                    ir = frames.first(rs.stream.infrared)
                    processed_frame = ir
                else:
                    aligned_frames = rs.align(rs.stream.color).process(frames)
                    processed_frame = aligned_frames.get_color_frame()

                if not processed_frame:
                    continue
            
                # convert frame to cv2
                color_image = np.asanyarray(processed_frame.get_data())
                # gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # detect AprilTag
            if not args.skip_detect:
                corners, ids, rejected = detector.detectMarkers(color_image)
            else:
                corners, ids, rejected = None, None, None

            if ids is not None:
                # estimate pose
                corners = np.array(corners)
                for i in range(len(ids)):
                    tag_id = ids[i][0]
                    corners_single_tag = corners[i][0]

                    # estimatePoseSingleMarkers is deprecated after some version and may cause issues on some hardware, use solvePnP instead
                    # rvec, tvec, objpoints = cv2.aruco.estimatePoseSingleMarkers(
                    #     corners[i], args.tag_size, camera_matrix, dist_coeffs
                    # )

                    success, rvec, tvec = cv2.solvePnP(
                        obj_points,           # 3D object points
                        corners_single_tag,   # 2D image points
                        camera_matrix,        # camera matrix
                        dist_coeffs,          # distortion coefficients
                        flags=cv2.SOLVEPNP_IPPE_SQUARE  # algorithm suitable for planar markers
                    )

                    if success:

                        if tag_id == args.left_hand_tag_id:
                            left_hand_pos = calculate_target_from_tag(corners_single_tag, rvec, tvec, T_cam_in_robot, T_target_in_tag)
                            if args.unit_test:
                                print(f"Left hand tag detected. ID: {tag_id}, Target: {left_hand_pos}")
                            else:
                                states_get('left_target_real_time')[:] = left_hand_pos
                            
                            if not headless:
                                color_image = draw_pose(
                                    color_image, corners_single_tag, 
                                    rvec, tvec, 
                                    camera_matrix, dist_coeffs, args.tag_size, tag_id,
                                    left_hand_pos,
                                )
                            
                        elif tag_id == args.right_hand_tag_id:
                            right_hand_pos = calculate_target_from_tag(corners_single_tag, rvec, tvec, T_cam_in_robot, T_target_in_tag)
                            if args.unit_test:
                                print(f"Right hand tag detected. ID: {tag_id}, Target: {right_hand_pos}")
                            else:
                                states_get('right_target_real_time')[:] = right_hand_pos

                            if not headless:
                                color_image = draw_pose(
                                    color_image, corners_single_tag, 
                                    rvec, tvec, 
                                    camera_matrix, dist_coeffs, args.tag_size, tag_id,
                                    right_hand_pos,
                                )
                        else:
                            if not headless:
                                color_image = draw_pose(
                                    color_image, corners_single_tag, 
                                    rvec, tvec, 
                                    camera_matrix, dist_coeffs, args.tag_size, tag_id,
                                )
                    else:
                        print(f"Pose estimation failed for tag ID: {tag_id}")

            # calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            if not headless:
                cv2.putText(color_image, f"FPS: {fps:.1f}", (10, color_image.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                print(f"FPS: {fps:.1f}")
            
            # handle display window
            if not headless:
                cv2.imshow('AprilTag Detection', color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


    finally:
        if not args.fake_camera:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()