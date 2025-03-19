# yapf: disable
import numpy as np


KP_DEFAULT = [
    185.256, 185.256, 185.256, 55.958,  # left leg
    185.256, 185.256, 185.256, 55.958,  # right leg
]
KD_DEFAULT = [
    9.2628, 9.2628, 9.2628, 2.7979,  # left leg
    9.2628, 9.2628, 9.2628, 2.7979,  # right leg
]
KP_2 = [
    1.1667, 1.1667, 1.1667, 0.5556,  # left leg
    1.1667, 1.1667, 1.1667, 0.5556,  # right leg
]
KD_2 = [
    0.0394, 0.0394, 0.0394, 0.0311,  # left leg
    0.0394, 0.0394, 0.0394, 0.0311,  # right leg
]
Q_CTRL_MIN = [
    -0.35, -0.52, -1.5, -1.9,
    -1.1, -0.52, -1.5, -1.9,
]
Q_CTRL_MAX = [
    1.1, 1., 0.74, 1.1,
    0.35, 1., 0.74, 1.1,
]
QD_LIMIT = [
    14.738, 12.356, 12.356, 16.747,
    14.738, 12.356, 12.356, 16.747,
]
TAU_LIMIT = [
    54., 95., 95., 30.,
    54., 95., 95., 30.,
]
DOF_NAMES = [
    'left_hip_joint',
    'left_thigh_joint',
    'left_shank_joint',
    'left_foot_joint',

    'right_hip_joint',
    'right_thigh_joint',
    'right_shank_joint',
    'right_foot_joint',
]
DOF_NAMES_2 = [
    'left_hip_roll_joint',
    'left_hip_pitch_joint',
    'left_knee_pitch_joint',
    'left_ankle_pitch_joint',

    'right_hip_roll_joint',
    'right_hip_pitch_joint',
    'right_knee_pitch_joint',
    'right_ankle_pitch_joint',
]

Q_CTRL_MIN = np.array(Q_CTRL_MIN)
Q_CTRL_MAX = np.array(Q_CTRL_MAX)
QD_LIMIT = np.array(QD_LIMIT)
TAU_LIMIT = np.array(TAU_LIMIT)
KP_DEFAULT = np.array(KP_DEFAULT)
KD_DEFAULT = np.array(KD_DEFAULT)
NUM_DOFS = 8

DOF_PRESETS = {
    'lleg': [
        'left_hip_joint',
        'left_thigh_joint',
        'left_shank_joint',
        'left_foot_joint',
    ],
    'rleg': [
        'right_hip_joint',
        'right_thigh_joint',
        'right_shank_joint',
        'right_foot_joint',
    ],
}

URDF = 'PTR1T1/urdf/PTR1T1_1.urdf'

FSA_IPS = [
    # left leg
    "192.168.137.70", "192.168.137.71", "192.168.137.72", "192.168.137.73",
    # right leg
    "192.168.137.50", "192.168.137.51", "192.168.137.52", "192.168.137.53",
]

FSE_IPS = []

FSA_SIGN = [
    1., 1., -1., -1.,
    1., -1., 1., 1.,
]

FSA_POSITION_KP = [
    1.1667, 1.1667, 1.1667, 0.5556,  # left leg
    1.1667, 1.1667, 1.1667, 0.5556,  # right leg
]

FSA_VELOCITY_KP = [
    0.0394, 0.0394, 0.0394, 0.0311,  # left leg
    0.0394, 0.0394, 0.0394, 0.0311,  # right leg
]
