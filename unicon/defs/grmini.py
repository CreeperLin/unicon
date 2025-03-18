# yapf: disable
import numpy as np


KP_DEFAULT = [
    185.256, 114.209, 114.209, 185.256, 55.958, 55.958,  # left leg
    185.256, 114.209, 114.209, 185.256, 55.958, 55.958,  # right leg
    114.209,  # waist
    114.209, 55.958, 55.958, 55.958, 55.958,  # left arm
    114.209, 55.958, 55.958, 55.958, 55.958,  # right arm
]
KD_DEFAULT = [
    9.2628, 5.71045, 5.71045, 9.2628, 2.7979, 2.7979,  # left leg
    9.2628, 5.71045, 5.71045, 9.2628, 2.7979, 2.7979,  # right leg
    5.71045,  # waist
    5.71045, 2.7979, 2.7979, 2.7979, 2.7979,  # left arm
    5.71045, 2.7979, 2.7979, 2.7979, 2.7979,  # right arm
]
KP_2 = [
    # left leg
    180.0, 120.0, 90.0, 120.0, 45.0, 45.0,
    # right leg
    180.0, 120.0, 90.0, 120.0, 45.0, 45.0,
    # waist
    90.0,
    # left arm
    90.0, 45.0, 45.0, 45.0, 45.0,
    # right arm
    90.0, 45.0, 45.0, 45.0, 45.0,
]
KD_2 = [
    # left leg
    10.0, 10.0, 8.0, 8.0, 2.5, 2.5,
    # right leg
    10.0, 10.0, 8.0, 8.0, 2.5, 2.5,
    # waist
    8.0,
    # left arm
    8.0, 2.5, 2.5, 2.5, 2.5,
    # right arm
    8.0, 2.5, 2.5, 2.5, 2.5,
]
Q_CTRL_MIN = [
    -2.617, -0.261, -2.617, -0.0872, -0.436, -0.436,
    -2.617, -1.57, -2.617, -0.0872, -0.436, -0.436,
    -2.617,
    -2.966, -0.174, -1.832, -0.349, -1.832,
    -2.966, -2.792, -1.832, -0.349, -1.832,
]
Q_CTRL_MAX = [
    2.617, 1.57, 2.617, 2.356, 0.436, 0.436,
    2.617, 0.261, 2.617, 2.356, 0.436, 0.436,
    2.617,
    2.966, 2.792, 1.832, 1.658, 1.832,
    2.966, 0.174, 1.832, 1.658, 1.832,
]
QD_LIMIT = [
    12.356, 14.738, 14.738, 12.356, 16.747, 16.747,
    12.356, 14.738, 14.738, 12.356, 16.747, 16.747,
    14.738,
    14.738, 16.747, 16.747, 16.747, 16.747,
    14.738, 16.747, 16.747, 16.747, 16.747
]
TAU_LIMIT = [
    # 95., 54., 54., 95., 30., 30.,
    # 95., 54., 54., 95., 30., 30.,
    135., 80., 80., 135., 50., 50.,
    135., 80., 80., 135., 50., 50.,
    54.,
    54., 30., 30., 30., 30.,
    54., 30., 30., 30., 30.,
]
DOF_NAMES = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_pitch_joint',
    'left_ankle_roll_joint',
    'left_ankle_pitch_joint',

    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_pitch_joint',
    'right_ankle_roll_joint',
    'right_ankle_pitch_joint',

    'waist_yaw_joint',

    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_pitch_joint',
    'left_wrist_yaw_joint',

    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_pitch_joint',
    'right_wrist_yaw_joint',
]
DOF_NAMES_2 = [
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

    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_yaw_joint',

    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_yaw_joint',
]

Q_CTRL_MIN = np.array(Q_CTRL_MIN)
Q_CTRL_MAX = np.array(Q_CTRL_MAX)
QD_LIMIT = np.array(QD_LIMIT)
TAU_LIMIT = np.array(TAU_LIMIT)
KP_DEFAULT = np.array(KP_DEFAULT)
KD_DEFAULT = np.array(KD_DEFAULT)
NUM_DOFS = 23
DOF_MAPS = {
    'left_leg': range(0, 6),
    'right_leg': range(6, 12),
    'legs': range(0, 12),
}

DOF_PRESETS = {
    'lower': [
        'left_hip_pitch_joint',
        'left_hip_roll_joint',
        'left_hip_yaw_joint',
        'left_knee_pitch_joint',
        'left_ankle_roll_joint',
        'left_ankle_pitch_joint',
        'right_hip_pitch_joint',
        'right_hip_roll_joint',
        'right_hip_yaw_joint',
        'right_knee_pitch_joint',
        'right_ankle_roll_joint',
        'right_ankle_pitch_joint',
    ],
}

URDF = 'GRMini1T1/urdf/GRMini1T1_full.urdf'

FSA_IPS = [
    # left leg
    "192.168.137.70", "192.168.137.71", "192.168.137.72", "192.168.137.73", "192.168.137.74", "192.168.137.75",
    # right leg
    "192.168.137.50", "192.168.137.51", "192.168.137.52", "192.168.137.53", "192.168.137.54", "192.168.137.55",
    # waist
    "192.168.137.90",
    # left arm
    "192.168.137.10", "192.168.137.11", "192.168.137.12", "192.168.137.13",
    "192.168.137.14",
    # right arm
    "192.168.137.30", "192.168.137.31", "192.168.137.32", "192.168.137.33",
    "192.168.137.34",
]


FSE_IPS = []


FSA_SIGN = [
    -1., +1., +1., -1., -1., +1.,
    +1., +1., +1., +1., -1., -1.,
    -1.,
    -1., -1., +1., -1., +1.,
    +1., -1., +1., +1., +1.,
]

FSA_POSITION_KP = [
    1.1667, 0.5556, 0.5556, 1.1667, 0.5556, 0.5556,  # left leg
    1.1667, 0.5556, 0.5556, 1.1667, 0.5556, 0.5556,  # right leg
    0.5556,  # waist
    0.5556, 0.5556, 0.5556, 0.5556, 0.5556,  # left arm
    0.5556, 0.5556, 0.5556, 0.5556, 0.5556,  # right arm
]

FSA_VELOCITY_KP = [
    0.0394, 0.0359, 0.0359, 0.0394, 0.0311, 0.0311,  # left leg
    0.0394, 0.0359, 0.0359, 0.0394, 0.0311, 0.0311,  # right leg
    0.0359,  # waist
    0.0359, 0.0311, 0.0311, 0.0311, 0.0311,  # left arm
    0.0359, 0.0311, 0.0311, 0.0311, 0.0311,  # right arm
]
