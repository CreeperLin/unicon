# yapf: disable

NAME = 'go2'
NUM_DOFS = 12
KP = [60.] * NUM_DOFS
KD = [5.] * NUM_DOFS

DOF_NAMES = [
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
]
DOF_NAMES_STD = {
    'FL_hip_joint': 'left_shoulder_yaw',
    'RL_hip_joint': 'left_hip_yaw',
    'FR_hip_joint': 'right_shoulder_yaw',
    'RR_hip_joint': 'right_hip_yaw',
    'FL_thigh_joint': 'left_shoulder_pitch',
    'RL_thigh_joint': 'left_hip_pitch',
    'FR_thigh_joint': 'right_shoulder_pitch',
    'RR_thigh_joint': 'right_hip_pitch',
    'FL_calf_joint': 'left_elbow_pitch',
    'RL_calf_joint': 'left_knee_pitch',
    'FR_calf_joint': 'right_elbow_pitch',
    'RR_calf_joint': 'right_knee_pitch',
}

Q_RESET = [
    0, 0.8, -1.5, 0, 0.8, -1.5,
    0, 1.0, -1.5, 0, 1.0, -1.5,
]
Q_BOOT = [
    0, 1.1, -2.7, 0, 1.1, -2.7,
    0, 1.1, -2.7, 0, 1.1, -2.7,
]

URDF = 'go2/urdf/go2.urdf'

ASSET_OPTIONS = {
    'flip_visual_attachments': True,
}
