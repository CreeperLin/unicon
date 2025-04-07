# yapf: disable
import numpy as np

a1_Hip_max = 0.802
a1_Hip_min = -0.802
a1_Thigh_max = 4.19
a1_Thigh_min = -1.05
a1_Calf_max = -0.916
a1_Calf_min = -2.7

NUM_DOFS = 12
Q_RESET = [
    0, 0.8, -1.5, 0, 0.8, -1.5,
    0, 1.0, -1.5, 0, 1.0, -1.5,
]
Q_BOOT = [
    0, 1.1, -2.7, 0, 1.1, -2.7,
    0, 1.1, -2.7, 0, 1.1, -2.7,
]
TAU_GRAVITY_COMP = [-0.65, 0, 0, +0.65, 0, 0, -0.65, 0, 0, +0.65, 0, 0]
# [-0.802, -1.05, -2.7, -0.802, -1.05, -2.7, -0.802, -1.05, -2.7, -0.802, -1.05, -2.7]
Q_CTRL_MIN = [a1_Hip_min, a1_Thigh_min, a1_Calf_min] * 4
# [0.802, 4.19, -0.916, 0.802, 4.19, -0.916, 0.802, 4.19, -0.916, 0.802, 4.19, -0.916]
Q_CTRL_MAX = [a1_Hip_max, a1_Thigh_max, a1_Calf_max] * 4
TAU_LIMIT = [20.0, 55.0, 55.0, 20.0, 55.0, 55.0, 20.0, 55.0, 55.0, 20.0, 55.0, 55.0]

Q_BOOT = np.array(Q_BOOT)
Q_RESET = np.array(Q_RESET)
Q_LIMIT = np.array([Q_CTRL_MIN, Q_CTRL_MAX])
TAU_LIMIT = np.array(TAU_LIMIT)
TAU_LIMIT = np.array([-TAU_LIMIT, TAU_LIMIT])
Q_CTRL_MIN = np.array(Q_CTRL_MIN)
Q_CTRL_MAX = np.array(Q_CTRL_MAX)

KP = [28.] * NUM_DOFS
KD = [1.] * NUM_DOFS
KP = np.array(KP)
KD = np.array(KD)

DOF_NAMES_2 = [
    'FR_hip', 'FR_thigh', 'FR_calf',
    'FL_hip', 'FL_thigh', 'FL_calf',
    'RR_hip', 'RR_thigh', 'RR_calf',
    'RL_hip', 'RL_thigh', 'RL_calf',
]
DOF_NAMES = [
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
]
NUM_DOFS = 12
DOF_MAPS = {
}

QD_LIMIT = [
    52.4, 28.6, 28.6,
] * 4
QD_LIMIT = np.array(QD_LIMIT)
TAU_LIMIT = [
    20., 55., 55.
] * 4
TAU_LIMIT = np.array(TAU_LIMIT)

URDF = 'a1/urdf/a1.urdf'
# URDF = 'a1/urdf/a1_bip.urdf'
