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

    ROOT_FRAME = 'pelvis'
    TARGET_FRAME = 'd435_link'
    URDF_PATH = ''

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    
    # 创建环境
    env_spacing = 2.0
    env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)
    
    # 加载机器人URDF
    asset_root = "/home/caojiahang/Code/G1/resources/robots/g1_description"
    robot_asset_file = "g1_29dof_fixed_hand_rev_5.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    robot_asset = gym.load_asset(sim, asset_root, robot_asset_file, asset_options)
    
    # 创建机器人Actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 1.0)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    robot_handle = gym.create_actor(env, robot_asset, pose, "robot", 0, 0)
    
    # 运行模拟
    while True:
        # 模拟步进
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # 获取相机变换矩阵
        T_cam_to_robot = get_camera_transform_in_simulation(
            gym, sim, env, robot_handle, TARGET_FRAME, ROOT_FRAME
        )
        
        # 使用变换矩阵进行后续处理...
        print("当前相机变换矩阵:\n", T_cam_to_robot)
        
        # 渲染等操作...

if __name__ == "__main__":
    simulate()
