# yapf: disable
import numpy as np


DOF_NAMES = [
    'left_hip_yaw_joint',
    'left_hip_roll_joint',
    'left_hip_pitch_joint',
    'left_knee_joint',
    'left_ankle_joint',
    'right_hip_yaw_joint',
    'right_hip_roll_joint',
    'right_hip_pitch_joint',
    'right_knee_joint',
    'right_ankle_joint',
    'torso_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
]
NUM_DOFS = 19

KP_DEFAULT = [
    200., 200., 200., 300., 40.,
    200., 200., 200., 300., 40.,
    300.,  # waist
    20., 20., 20., 20.,
    20., 20., 20., 20.,
]
KD_DEFAULT = [
    5., 5., 5., 6., 2.,
    5., 5., 5., 6., 2.,
    6.,  # waist
    0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5,
]
Q_CTRL_MIN = [
    -0.43, -0.43, -1.57, -0.26, -0.87,
    -0.43, -0.43, -1.57, -0.26, -0.87,
    -2.35,
    -2.87, -0.34, -1.3, -1.25,
    -2.87, -3.11, -4.45, -1.25,
]
Q_CTRL_MAX = [
    0.43, 0.43, 1.57, 2.05, 0.52,
    0.43, 0.43, 1.57, 2.05, 0.52,
    2.35,
    2.87, 3.11, 4.45, 2.61,
    2.87, 0.34, 1.3, 2.61,
]
QD_LIMIT = [
    23., 23., 23., 14., 9.,
    23., 23., 23., 14., 9.,
    23.,
    9., 9., 20., 20.,
    9., 9., 20., 20.,
]
TAU_LIMIT = [
    200., 200., 200., 300., 40.,
    200., 200., 200., 300., 40.,
    200.,
    40., 40., 18., 18.,
    40., 40., 18., 18.,
]
Q_CTRL_MIN = np.array(Q_CTRL_MIN)
Q_CTRL_MAX = np.array(Q_CTRL_MAX)
QD_LIMIT = np.array(QD_LIMIT)
TAU_LIMIT = np.array(TAU_LIMIT)
KP_DEFAULT = np.array(KP_DEFAULT)
KD_DEFAULT = np.array(KD_DEFAULT)
