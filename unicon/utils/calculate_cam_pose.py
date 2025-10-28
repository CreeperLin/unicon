import os

import numpy as np
from scipy.spatial.transform import Rotation as R

import isaacgym
from isaacgym import gymapi



def get_camera_transform_in_simulation(gym, sim, env, robot_handle, camera_link_name, root_link_name):
    """
    在模拟运行时获取相机到机器人基座的变换矩阵
    
    参数:
        gym: Isaac Gym 实例
        sim: 模拟实例
        env: 环境实例
        robot_handle: 机器人Actor句柄
        camera_link_name: 相机连杆名称
        
    返回:
        T_cam_to_robot: 4x4齐次变换矩阵
    """
    # 获取刚体状态
    rigid_body_states = gym.get_actor_rigid_body_states(env, robot_handle, gymapi.STATE_POS)
    
    # 获取刚体索引字典
    rigid_body_dict = gym.get_actor_rigid_body_dict(env, robot_handle)
    
    # 检查相机连杆是否存在
    if camera_link_name not in rigid_body_dict:
        raise ValueError(f"找不到相机连杆: {camera_link_name}")
    
    # 获取相机连杆索引
    camera_index = rigid_body_dict[camera_link_name]
    
    # 获取相机位姿
    camera_pose = rigid_body_states['pose'][camera_index]
    
    # 获取基座连杆位姿
    base_index = rigid_body_dict[root_link_name]
    base_pose = rigid_body_states['pose'][base_index]
    
    # 将位姿转换为变换矩阵
    def pose_to_matrix(pose):
        T = np.eye(4)
        T[:3, 3] = [pose['p']['x'], pose['p']['y'], pose['p']['z']]
        rot = R.from_quat([
            pose['r']['x'], 
            pose['r']['y'], 
            pose['r']['z'], 
            pose['r']['w']
        ])
        T[:3, :3] = rot.as_matrix()
        return T
    
    # 计算相机相对于基座的变换
    T_cam_to_world = pose_to_matrix(camera_pose)
    T_base_to_world = pose_to_matrix(base_pose)
    T_world_to_base = np.linalg.inv(T_base_to_world)
    T_cam_to_robot = T_world_to_base @ T_cam_to_world
    
    return T_cam_to_robot

# 在模拟循环中使用
def simulate():

    # ROOT_FRAME = 'imu_in_torso'
    # TARGET_FRAME = 'd435_link'
    # asset_root = "/home/caojiahang/Code/G1/resources/robots/g1_description"
    # robot_asset_file = "g1_29dof_fixed_hand_rev_5.urdf"

    ROOT_FRAME = 'Trunk'
    TARGET_FRAME = 'right_hand_end_link'
    asset_root = "/home/caojiahang/Code/G1/resources/robots/Booster-T1"
    robot_asset_file = "T1_serial_simple_version.urdf"

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z  # 设置z轴为全局坐标系向上
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    
    # 创建环境
    env_spacing = 2.0
    # z轴向上，y轴为左右，z轴为高度
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)
    
    # 加载机器人URDF
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    robot_asset = gym.load_asset(sim, asset_root, robot_asset_file, asset_options)
    
    # 创建机器人Actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 1.0)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    robot_handle = gym.create_actor(env, robot_asset, pose, "robot", 0, 0)
    
    # 获取机器人关节数量
    num_dofs = gym.get_asset_dof_count(robot_asset)
    # 设定默认关节角度（可根据实际机器人修改）
    # default_joint = np.zeros(num_dofs, dtype=np.float32)  # 这里假设默认角度为0
    # 如果有具体的default_joint值，可以替换上面一行
    default_joint = np.array([
        0.0000,  0.0000,  
        0.0000, -1.5000,  0.0000, -1.6000,  
        0.0000,  1.5000,  0.0000,  1.6000,  
        0.0000, 
        -0.2500,  0.0000,  0.0000,  0.5500, -0.2900, 0.0000, 
        -0.2500,  0.0000,  0.0000,  0.5500, -0.2900, 0.0000
    ], dtype=np.float32)

    def set_camera(position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # 创建 viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    set_camera([2, 2, 2], [0, 0, 1])

    while not gym.query_viewer_has_closed(viewer):
        # 每步直接修改机器人关节状态
        dof_states = gym.get_actor_dof_states(env, robot_handle, gymapi.STATE_ALL)
        dof_states[0:num_dofs] = default_joint  # 位置
        dof_states[num_dofs:] = 0.0            # 速度
        gym.set_actor_dof_states(env, robot_handle, dof_states, gymapi.STATE_ALL)

        # 模拟步进
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # 渲染
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

        # 获取相机变换矩阵
        T_cam_to_robot = get_camera_transform_in_simulation(
            gym, sim, env, robot_handle, TARGET_FRAME, ROOT_FRAME
        )

        # 使用变换矩阵进行后续处理...
        matrix_str = np.array2string(T_cam_to_robot, formatter={'float_kind':lambda x: "%.6f," % x})
        print(f"当前{TARGET_FRAME}到{ROOT_FRAME}变换矩阵:\n", matrix_str)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    simulate()
