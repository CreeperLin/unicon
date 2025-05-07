# yapf: disable
import numpy as np


NAME = 'gr1t2'
_gr1t2_joint_pd_control_kp = [
    251.625, 362.5214, 200, 200, 10.9805, 0.25,
    251.625, 362.5214, 200, 200, 10.9805, 0.25,
    362.5214, 362.5214, 362.5214,
    10.0, 10.0, 10.0,
    92.85, 92.85, 112.06, 112.06, 10.0, 10.0, 10.0,
    92.85, 92.85, 112.06, 112.06, 10.0, 10.0, 10.0,
]
_gr1t2_joint_pd_control_kd = [
    14.72, 10.0833, 11, 11, 0.5991, 0.01,
    14.72, 10.0833, 11, 11, 0.5991, 0.01,
    10.0833, 10.0833, 10.0833,
    1, 1, 1,
    2.575, 2.575, 3.1, 3.1, 1.0, 1.0, 1.0,
    2.575, 2.575, 3.1, 3.1, 1.0, 1.0, 1.0,
]
_gr1t2_joint_limit_lower = [
    -0.09, -0.7, -1.75, -0.09, -1.05, -0.44,
    -0.79, -0.7, -1.75, -0.09, -1.05, -0.44,
    -1.05, -0.52, -0.7,
    -0.87, -0.35, -2.71,
    -2.79, -0.57, -2.97, -2.27, -2.97, -0.96, -0.61,
    -2.79, -3.27, -2.97, -2.27, -2.97, -0.87, -0.61
]
_gr1t2_joint_limit_upper = [
    0.79, 0.7, 0.7, 1.92, 0.52, 0.44,
    0.09, 0.7, 0.7, 1.92, 0.52, 0.44,
    1.05, 1.22, 0.7,
    0.87, 0.35, 2.71,
    1.92, 3.27, 2.97, 2.27, 2.97, 0.87, 0.61,
    1.92, 0.57, 2.97, 2.27, 2.97, 0.96, 0.61,
]
_gr1t2_joint_limit_lower2 = [
    -0.09, -0.7, -1.75, -0.09, -1.05, -0.44,
    -0.09, -0.7, -1.75, -0.09, -1.05, -0.44,
    -1.05, -0.52, -0.7,
    -2.71, -0.35, -0.52,
    -2.79, -0.57, -2.97, -2.27, -2.97, -0.61, -0.61,
    -2.79, -0.57, -2.97, -2.27, -2.97, -0.61, -0.61
]
_gr1t2_joint_limit_upper2 = [
    0.79, 0.7, 0.7, 1.92, 0.52, 0.44,
    0.09, 0.7, 0.7, 1.92, 0.52, 0.44,
    1.05, 1.22, 0.7,
    2.71, 0.35, 0.35,
    1.92, 3.27, 2.97, 2.27, 2.97, 0.61, 0.61,
    1.92, 0.57, 2.97, 2.27, 2.97, 0.61, 0.61,
]
_gr1t2_joint_reset_pos = [
    0.0, 0.0, -0.4, 0.8, -0.4, 0.0,
    0.0, 0.0, -0.4, 0.8, -0.4, 0.0,
    0.0, -0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.2, 0.0, -0.3, 0.0, 0.0, 0.0,
    0.0, -0.2, 0.0, -0.3, 0.0, 0.0, 0.0,
]
_gr1t2_joint_velocity_limit = [
    12.15, 16.76, 37.38, 37.38, 20.32, 20.32,
    12.15, 16.76, 37.38, 37.38, 20.32, 20.32,
    16.76, 16.76, 16.76,
    27.96, 27.96, 27.96,
    9.11, 9.11, 7.33, 7.33, 24.4, 27.96, 27.96,
    9.11, 9.11, 7.33, 7.33, 24.4, 27.96, 27.96,
]
_gr1t2_joint_torque_limit = [
    48.0, 66.0, 225.0, 225.0, 15.0, 30.0,
    48.0, 66.0, 225.0, 225.0, 15.0, 30.0,
    66.0, 66.0, 66.0,
    3.95, 3.95, 3.95,
    38.0, 38.0, 30.0, 30.0, 10.2, 3.95, 3.95,
    38.0, 38.0, 30.0, 30.0, 10.2, 3.95, 3.95,
]
_gr1t2_joint_names = [
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_hip_pitch_joint',
    'left_knee_pitch_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',

    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_hip_pitch_joint',
    'right_knee_pitch_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',

    'waist_yaw_joint',
    'waist_pitch_joint',
    'waist_roll_joint',

    # 'head_roll_joint',
    'head_pitch_joint',
    'head_roll_joint',
    'head_yaw_joint',

    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_pitch_joint',
    'left_wrist_yaw_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    # 'left_end_effector_joint',

    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_pitch_joint',
    'right_wrist_yaw_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    # 'right_end_effector_joint',
]
_gr1t2_joint_names_2 = [
    'l_hip_roll',
    'l_hip_yaw',
    'l_hip_pitch',
    'l_knee_pitch',
    'l_ankle_pitch',
    'l_ankle_roll',
    'r_hip_roll',
    'r_hip_yaw',
    'r_hip_pitch',
    'r_knee_pitch',
    'r_ankle_pitch',
    'r_ankle_roll',
    'joint_waist_yaw',
    'joint_waist_pitch',
    'joint_waist_roll',
    'joint_head_pitch',
    'joint_head_roll',
    'joint_head_yaw',
    'l_shoulder_pitch',
    'l_shoulder_roll',
    'l_shoulder_yaw',
    'l_elbow_pitch',
    'l_wrist_yaw',
    'l_wrist_roll',
    'l_wrist_pitch',
    'r_shoulder_pitch',
    'r_shoulder_roll',
    'r_shoulder_yaw',
    'r_elbow_pitch',
    'r_wrist_yaw',
    'r_wrist_roll',
    'r_wrist_pitch',
]


Q_CTRL_MIN = np.array(_gr1t2_joint_limit_lower)
Q_CTRL_MAX = np.array(_gr1t2_joint_limit_upper)
Q_RESET = np.array(_gr1t2_joint_reset_pos)
QD_LIMIT = np.array(_gr1t2_joint_velocity_limit)
TAU_LIMIT = np.array(_gr1t2_joint_torque_limit)
KP = np.array(_gr1t2_joint_pd_control_kp)
KD = np.array(_gr1t2_joint_pd_control_kd)
NUM_DOFS = 32
DOF_NAMES = _gr1t2_joint_names
DOF_NAMES_2 = _gr1t2_joint_names_2
DOF_MAPS = {
    'left_leg': range(0, 6),
    'right_leg': range(6, 12),
    'legs': range(0, 12),
    'waist': range(12, 15),
    'head': range(15, 18),
    'left_arm': range(18, 25),
    'right_arm': range(25, 32),
}

DOF_PRESETS = {
    'lower': [
        'left_hip_roll_joint',
        'left_hip_yaw_joint',
        'left_hip_pitch_joint',
        'left_knee_pitch_joint',
        'left_ankle_pitch_joint',
        'left_ankle_roll_joint',
        'right_hip_roll_joint',
        'right_hip_yaw_joint',
        'right_hip_pitch_joint',
        'right_knee_pitch_joint',
        'right_ankle_pitch_joint',
        'right_ankle_roll_joint',
    ],
    'reduced': [
        'left_hip_roll_joint',
        'left_hip_yaw_joint',
        'left_hip_pitch_joint',
        'left_knee_pitch_joint',
        'left_ankle_pitch_joint',
        'left_ankle_roll_joint',
        'right_hip_roll_joint',
        'right_hip_yaw_joint',
        'right_hip_pitch_joint',
        'right_knee_pitch_joint',
        'right_ankle_pitch_joint',
        'right_ankle_roll_joint',
        'waist_yaw_joint',
        'waist_pitch_joint',
        'waist_roll_joint',
        'left_shoulder_pitch_joint',
        'left_shoulder_roll_joint',
        'left_shoulder_yaw_joint',
        'left_elbow_pitch_joint',
        'right_shoulder_pitch_joint',
        'right_shoulder_roll_joint',
        'right_shoulder_yaw_joint',
        'right_elbow_pitch_joint',
    ],
    'lleg': [
        'left_hip_roll_joint',
        'left_hip_yaw_joint',
        'left_hip_pitch_joint',
        'left_knee_pitch_joint',
        'left_ankle_pitch_joint',
        'left_ankle_roll_joint',
    ],
}

URDF = 'gr2/urdf/GR1T2.urdf'

FSA_IPS = [
    # left leg
    "192.168.137.70", "192.168.137.71", "192.168.137.72", "192.168.137.73", "192.168.137.74", "192.168.137.75",
    # right leg
    "192.168.137.50", "192.168.137.51", "192.168.137.52", "192.168.137.53", "192.168.137.54", "192.168.137.55",
    # waist
    "192.168.137.90", "192.168.137.91", "192.168.137.92",
    # head
    "192.168.137.93", "192.168.137.94", "192.168.137.95",
    # left arm
    "192.168.137.10", "192.168.137.11", "192.168.137.12", "192.168.137.13",
    "192.168.137.14", "192.168.137.15", "192.168.137.16",
    # right arm
    "192.168.137.30", "192.168.137.31", "192.168.137.32", "192.168.137.33",
    "192.168.137.34", "192.168.137.35", "192.168.137.36",
]


FSE_IPS = [
    # left leg
    "192.168.137.170", "192.168.137.171", "192.168.137.172", "192.168.137.173", "192.168.137.174", "192.168.137.175",
    # right leg
    "192.168.137.150", "192.168.137.151", "192.168.137.152", "192.168.137.153", "192.168.137.154", "192.168.137.155",
    # waist (ypr)
    "192.168.137.190", "192.168.137.191", "192.168.137.192",
]

FSA_SIGN = [
    -1.0, 1.0, 1.0, -1.0, -1.0, 1.0,  # left leg
    -1.0, 1.0, -1.0, 1.0, 1.0, -1.0,  # right leg
    -1.0, -1.0, -1.0,  # waist
    -1.0, -1.0, -1.0,  # head
    -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0,  # left arm
    1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right arm
]
FSE_SIGN = [
    1.0, 1.0, -1.0, 1.0, 1.0, -1.0,  # left leg
    1.0, 1.0, 1.0, -1.0, -1.0, 1.0,  # right leg
    1.0, 1.0, 1.0,  # waist
]
FSE_RATIO = [
    2.0, 2.77, 2.514, 1.0, 1.0, 1.0,  # left leg
    2.0, 2.77, 2.514, 1.0, 1.0, 1.0,  # right leg
    4.08, 1.0, 1.0,  # waist
]

FSE2FSA = {
    "192.168.137.170": "192.168.137.70",
    "192.168.137.171": "192.168.137.71",
    "192.168.137.172": "192.168.137.72",
    "192.168.137.173": "192.168.137.73",
    "192.168.137.174": "192.168.137.74",
    "192.168.137.175": "192.168.137.75",
    "192.168.137.150": "192.168.137.50",
    "192.168.137.151": "192.168.137.51",
    "192.168.137.152": "192.168.137.52",
    "192.168.137.153": "192.168.137.53",
    "192.168.137.154": "192.168.137.54",
    "192.168.137.155": "192.168.137.55",
    "192.168.137.190": "192.168.137.90",
    "192.168.137.191": "192.168.137.91",
    "192.168.137.192": "192.168.137.92",
}


FSA_POSITION_KP = [
    0.997, 1.023, 1.061, 1.061, 0.508, 0.508,
    0.997, 1.023, 1.061, 1.061, 0.508, 0.508,
    1.023, 1.023*5, 1.023*5,
    0.556, 0.556, 0.556,
    0.556, 0.556, 0.556, 0.556, 0, 0, 0,
    0.556, 0.556, 0.556, 0.556, 0, 0, 0,
]
FSA_VELOCITY_KP = [
    0.044, 0.03, 0.263, 0.263, 0.004, 0.004,
    0.044, 0.03, 0.263, 0.263, 0.004, 0.004,
    0.03, 0.03, 0.03,
    0.03, 0.03, 0.03,
    0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
    0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
]
