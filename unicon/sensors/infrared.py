import os
import argparse
import time

import numpy as np
import pyrealsense2 as rs
import cv2
from scipy.spatial.transform import Rotation as R

from unicon.states import states_get, states_init

# 可编辑参数
MAX_CANDIDATES = 10         # 选取最亮点的数量
XY_MIN_DIST = 0.05         # 两点最大xy距离（米）
CIRCLE_RADIUS = 1.0         # 圆周半径（米）
Z_CLIP = 0.2                # z轴clip范围（米）

def open_camera(width=640, height=480, fps=30, exposure=200, gain=32, laser_power=360):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)

    # 设置曝光和增益
    device = profile.get_device()
    sensors = device.query_sensors()
    for sensor in sensors:
        if sensor.get_info(rs.camera_info.name) == 'Stereo Module':
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.exposure, exposure)
            sensor.set_option(rs.option.gain, gain)
            sensor.set_option(rs.option.laser_power, laser_power)
            break

    # 获取内参
    infrared_profile = profile.get_stream(rs.stream.infrared)
    intrinsics = infrared_profile.as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                              [0, intrinsics.fy, intrinsics.ppy],
                              [0, 0, 1]])
    dist_coeffs = np.array(intrinsics.coeffs)
    return pipeline, camera_matrix, dist_coeffs, profile

def get_brightest_points(ir_img, num_points=MAX_CANDIDATES):
    # 归一化并高斯模糊
    blur = cv2.GaussianBlur(ir_img, (5, 5), 0)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(blur)
    # 阈值化，保留最亮的区域
    _, thresh = cv2.threshold(blur, maxVal-10, 255, cv2.THRESH_BINARY)
    # 连通域分析
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centers.append((cx, cy))
    # 按亮度排序
    centers = sorted(centers, key=lambda c: ir_img[c[1], c[0]], reverse=True)
    return centers[:num_points]

def get_3d_points(centers, depth_frame, intrinsics):
    points_3d = []
    for (cx, cy) in centers:
        depth = depth_frame.get_distance(cx, cy)
        if depth == 0:
            continue
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
        points_3d.append(np.array(point))
    return points_3d

def filter_and_assign(points_3d, xy_min_dist=XY_MIN_DIST, circle_radius=CIRCLE_RADIUS, z_clip=Z_CLIP):
    # 只保留两个点，且xy距离合理
    if len(points_3d) == 0:
        return None, None, -1, -1
    # 按x排序
    points_3d = sorted(points_3d, key=lambda p: p[0])
    p1 = points_3d[0]
    filtered = [p1]
    filtered_idx = [0]

    for j in range(1, len(points_3d)):
        dist = np.linalg.norm(p1[:2] - points_3d[j][:2])
        if dist < xy_min_dist:
            continue
        else:
            filtered.append(points_3d[j])
            filtered_idx.append(j)
            break

    clipped_filtered = []
    for p in filtered:
        dist = np.linalg.norm(p[:2])
        # 如果距离太远，投影到圆周上
        if dist > circle_radius:
            theta = np.arctan2(p[1], p[0])
            p[:2] = circle_radius * np.array([np.cos(theta), np.sin(theta)])
        # z裁剪
        p[2] = np.clip(p[2], -z_clip, z_clip)
        clipped_filtered.append(p)

    if len(clipped_filtered) == 1:
        p = clipped_filtered[0]
        if p[0] < 0:
            left, right, left_idx, right_idx = p, None, filtered_idx[0], -1
        else:
            left, right, left_idx, right_idx = None, p, -1, filtered_idx[0]

    # 按x排序，左为left，右为right
    if len(clipped_filtered) == 2:
        p1, p2 = clipped_filtered[0], clipped_filtered[1]
        if p1[0] < p2[0]:
            left, right, left_idx, right_idx = p1, p2, filtered_idx[0], filtered_idx[1]
        else:
            left, right, left_idx, right_idx = p2, p1, filtered_idx[1], filtered_idx[0]
    return left, right, left_idx, right_idx

def calculate_orientation(from_point, to_point):
    # 计算从机器人到球的方向，返回四元数
    direction = to_point - from_point
    direction = direction / np.linalg.norm(direction)
    # 坐标系定义：z轴朝上，x轴朝前
    x_axis = direction
    z_axis = np.array([0, 0, 1])
    y_axis = np.cross(z_axis, x_axis)
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = np.array([0, 1, 0])
    else:
        y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    rot_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)
    rot = R.from_matrix(rot_matrix)
    quat = rot.as_quat()
    return quat

def get_T_cam_in_robot():
    # 可根据实际标定修改
    T_camlink_in_robot = np.array([[0.674302, 0.000000, 0.738455, 0.097214],
                                   [-0.000000, 1.000000, -0.000000, 0.019770],
                                   [-0.738455, -0.000000, 0.674302, 0.281950],
                                   [0.000000, 0.000000, 0.000000, 1.000000]])
    Ry_pos = R.from_euler('y', 90, degrees=True).as_matrix()
    Rx_neg = R.from_euler('x', -90, degrees=True).as_matrix()
    T_cam_in_camlink = np.eye(4)
    T_cam_in_camlink[0:3, 0:3] = Rx_neg @ Ry_pos
    T_cam_in_robot = T_camlink_in_robot @ T_cam_in_camlink
    return T_cam_in_robot

def transform_to_robot(points_3d, T_cam_in_robot):
    points_robot = []
    for p in points_3d:
        p_h = np.concatenate([p, [1]])
        p_robot = T_cam_in_robot @ p_h
        points_robot.append(p_robot[:3])
    return points_robot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ut', '--unit_test', action='store_true', help='unit_test mode')
    parser.add_argument('-hdls', '--headless', action='store_true', help='run in headless mode')
    parser.add_argument('--exposure', type=int, default=200, help='infrared exposure')
    parser.add_argument('--gain', type=int, default=16, help='infrared gain')
    args = parser.parse_args()
    headless = args.headless

    if not args.unit_test:
        states_init(use_shm=True, load=True, reuse=True)
    else:
        print("Unit test mode: states not initialized")

    pipeline, camera_matrix, dist_coeffs, profile = open_camera(exposure=args.exposure, gain=args.gain)
    T_cam_in_robot = get_T_cam_in_robot()
    try:
        while True:
            start_time = time.time()
            frames = pipeline.wait_for_frames()
            ir_frame = frames.get_infrared_frame()
            depth_frame = frames.get_depth_frame()
            if not ir_frame or not depth_frame:
                print('Waiting for ir_frame and depth_frame')
                continue
            ir_img = np.asanyarray(ir_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            # 1. 预处理
            centers = get_brightest_points(ir_img)
            # 2. 取最亮点
            # 3. 计算3D坐标
            intrinsics = ir_frame.profile.as_video_stream_profile().get_intrinsics()
            points_3d_cam = get_3d_points(centers, depth_frame, intrinsics)
            # 4. 坐标筛选
            left, right, left_idx, right_idx = filter_and_assign(points_3d_cam)
            if left is not None or right is not None:
                # 初始化变量
                left_target = None
                right_target = None

                # 处理左点
                if left is not None:
                    left_robot = transform_to_robot([left], T_cam_in_robot)[0]
                    quat_left = calculate_orientation(np.zeros(3), left_robot)
                    left_target = np.concatenate([left_robot, quat_left])

                # 处理右点
                if right is not None:
                    right_robot = transform_to_robot([right], T_cam_in_robot)[0]
                    quat_right = calculate_orientation(np.zeros(3), right_robot)
                    right_target = np.concatenate([right_robot, quat_right])

                # 输出结果
                if args.unit_test:
                    print("Left:", left_target if left_target is not None else "None")
                    print("Right:", right_target if right_target is not None else "None")
                else:
                    # 只更新非None的目标
                    if left_target is not None:
                        states_get('left_target_real_time')[:] = left_target
                    if right_target is not None:
                        states_get('right_target_real_time')[:] = right_target
                # 7. 可视化
                if not headless:
                    vis_img = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)
                    for c in centers:
                        cv2.circle(vis_img, c, 5, (255, 0, 0), 2)
                    if left is not None:
                        p = left
                        px = int(p[0] * camera_matrix[0, 0] / p[2] + camera_matrix[0, 2])
                        py = int(p[1] * camera_matrix[1, 1] / p[2] + camera_matrix[1, 2])
                        cv2.circle(vis_img, (px, py), 8, (0, 255, 0), -1)
                        cv2.putText(vis_img, "Left", tuple(centers[left_idx]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    if right is not None:
                        p = right
                        px = int(p[0] * camera_matrix[0, 0] / p[2] + camera_matrix[0, 2])
                        py = int(p[1] * camera_matrix[1, 1] / p[2] + camera_matrix[1, 2])
                        cv2.circle(vis_img, (px, py), 8, (0, 255, 0), -1)
                        cv2.putText(vis_img, "Right", tuple(centers[right_idx]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                if not headless:
                    vis_img = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)
                    for c in centers:
                        cv2.circle(vis_img, c, 5, (0, 0, 255), 2)

            # 显示FPS
            fps = 1.0 / (time.time() - start_time)
            if not headless:
                cv2.putText(vis_img, f"FPS: {fps:.1f}", (10, vis_img.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                print(f"FPS: {fps:.1f}")
            
            # handle display window
            if not headless:
                cv2.imshow('IR Ball Detection', vis_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()