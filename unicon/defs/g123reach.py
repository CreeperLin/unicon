# yapf: disable
import numpy as np


NAME = 'g1'
KP = [
    150.0, 150.0, 150.0, 300.0, 40.0, 40.0,
    150.0, 150.0, 150.0, 300.0, 40.0, 40.0,
    300.0, 300.0, 300.0,
    10.0, 10.0, 10.0, 10.0, 10.0, 5.0, 5.0,
    10.0, 10.0, 10.0, 10.0, 10.0, 5.0, 5.0,
]
KD = [
    2., 2., 2., 4., 2., 2.,
    2., 2., 2., 4., 2., 2.,
    5., 5., 5.,
    0.25, 0.25, 0.25, 0.25, 0.25, 0.1, 0.1,
    0.25, 0.25, 0.25, 0.25, 0.25, 0.1, 0.1,
]
Q_CTRL_MIN = [
    -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
    -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
    -2.618, -0.52, -0.52,
    -3.0892, -1.5882, -2.618, -1.0472, -1.9722221, -1.6144296, -1.6144296,
    -3.0892, -2.2515, -2.618, -1.0472, -1.9722221, -1.6144296, -1.6144296,
]
Q_CTRL_MAX = [
    2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
    2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
    2.618, 0.52, 0.52,
    2.6704, 2.2515, 2.618, 2.0944, 1.9722221, 1.6144296, 1.6144296,
    2.6704, 1.5882, 2.618, 2.0944, 1.9722221, 1.6144296, 1.6144296,
]
QD_LIMIT = [
    32., 20., 32., 20., 37., 37.,
    32., 20., 32., 20., 37., 37.,
    32., 32., 32.,
    37., 37., 37., 37., 37., 37., 37.,
    37., 37., 37., 37., 37., 37., 37.,
]
TAU_LIMIT = [
    88., 139., 88., 139., 50., 50.,
    88., 139., 88., 139., 50., 50.,
    54., 54., 54.,
    54., 30., 30., 30., 30., 30., 30.,
    54., 30., 30., 30., 30., 30., 30.,
]
DOF_NAMES = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',

    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',

    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',

    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',

    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
]
DOF_NAMES_STD = {
    'left_hip_pitch_joint': 'left_hip_pitch',
    'left_hip_roll_joint': 'left_hip_roll',
    'left_hip_yaw_joint': 'left_hip_yaw',
    'left_knee_joint': 'left_knee_pitch',
    'left_ankle_pitch_joint': 'left_ankle_pitch',
    'left_ankle_roll_joint': 'left_ankle_roll',
    'right_hip_pitch_joint': 'right_hip_pitch',
    'right_hip_roll_joint': 'right_hip_roll',
    'right_hip_yaw_joint': 'right_hip_yaw',
    'right_knee_joint': 'right_knee_pitch',
    'right_ankle_pitch_joint': 'right_ankle_pitch',
    'right_ankle_roll_joint': 'right_ankle_roll',
    'waist_yaw_joint': 'waist_yaw',
    'waist_roll_joint': 'waist_roll',
    'waist_pitch_joint': 'waist_pitch',
    'left_shoulder_pitch_joint': 'left_shoulder_pitch',
    'left_shoulder_roll_joint': 'left_shoulder_roll',
    'left_shoulder_yaw_joint': 'left_shoulder_yaw',
    'left_elbow_joint': 'left_elbow_pitch',
    # 'left_wrist_roll_joint': 'left_wrist_roll',
    'left_wrist_roll_joint': 'left_wrist_yaw',
    'left_wrist_pitch_joint': 'left_wrist_pitch',
    # 'left_wrist_yaw_joint': 'left_wrist_yaw',
    'left_wrist_yaw_joint': 'left_wrist_roll',
    'right_shoulder_pitch_joint': 'right_shoulder_pitch',
    'right_shoulder_roll_joint': 'right_shoulder_roll',
    'right_shoulder_yaw_joint': 'right_shoulder_yaw',
    'right_elbow_joint': 'right_elbow_pitch',
    # 'right_wrist_roll_joint': 'right_wrist_roll',
    'right_wrist_roll_joint': 'right_wrist_yaw',
    'right_wrist_pitch_joint': 'right_wrist_pitch',
    # 'right_wrist_yaw_joint': 'right_wrist_yaw',
    'right_wrist_yaw_joint': 'right_wrist_roll',
}

Q_CTRL_MIN = np.array(Q_CTRL_MIN)
Q_CTRL_MAX = np.array(Q_CTRL_MAX)
QD_LIMIT = np.array(QD_LIMIT)
TAU_LIMIT = np.array(TAU_LIMIT)
KP = np.array(KP)
KD = np.array(KD)
NUM_DOFS = len(DOF_NAMES)
DOF_MAPS = {
    'left_leg': range(0, 6),
    'right_leg': range(6, 12),
    'legs': range(0, 12),
}

Q_RESET = [
    -0.1,  0. ,  0. ,  0.3, -0.2,  0. , 
    -0.1,  0. ,  0. ,  0.3, -0.2,  0. ,  
    0. ,  0. ,  0. ,  
    0. ,  0.2,  0. ,  0. ,  0. ,  0. ,  0. ,
    0. , -0.2,  0. ,  0. ,  0. ,  0. ,  0. 
]

# URDF = 'g1_description/g1_29dof_rev_1_0.urdf'
# URDF = 'g1/urdf/g1_29dof_boxcollision_fix_inertia_1.urdf'
URDF = "/home/caojiahang/Code/G1/resources/robots/g1_description/g1_23dof_rev_5.urdf"
# MJCF = "/home/caojiahang/Code/G1/resources/robots/g1_description/g1_29dof_fixed_hand_rev_5.xml"

# USE_SENSOR = ['imu_in_torso', 'imu_in_pelvis']
USE_SENSOR = ['imu_in_pelvis', 'imu_in_torso']
CUSTOM_EXTRA_OBS = [
    'rpy2',
    'ang_vel2',
    'left_target',
    'right_target',
    'reach_stage',
]
