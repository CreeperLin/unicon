csv_dof_names_h1_2 = [
    'left_hip_yaw_joint',
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_yaw_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'torso_joint',
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

csv_dof_names_g1 = [
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


def load(
    csv_path=None,
    robot_def=None,
):
    dof_names = getattr(robot_def, 'DOF_NAMES')
    q_min = getattr(robot_def, 'Q_CTRL_MIN')
    q_max = getattr(robot_def, 'Q_CTRL_MAX')
    import numpy as np
    csv_dof_names = csv_dof_names_g1
    csv_dof_remap = {
        'left_knee_pitch_joint': 'left_knee_joint',
        'right_knee_pitch_joint': 'right_knee_joint',
        # 'waist_yaw_joint': 'torso_joint',
        'left_elbow_pitch_joint': 'left_elbow_joint',
        'right_elbow_pitch_joint': 'right_elbow_joint',
    }
    print('csv_path', csv_path)
    csv_frames = np.genfromtxt(csv_path, delimiter=',')
    num_dofs = len(csv_dof_names)
    csv_q_w = np.ones(num_dofs)
    csv_q_b = np.zeros(num_dofs)
    # hip_b = 0.25
    # csv_q_b[csv_dof_names.index('left_hip_roll_joint')] = hip_b
    # csv_q_b[csv_dof_names.index('right_hip_roll_joint')] = -hip_b
    csv_dof_map = [csv_dof_names.index(csv_dof_remap.get(n, n)) for n in dof_names]
    print(num_dofs, csv_dof_names)
    print('csv_dof_map', csv_dof_map)
    print(csv_frames.shape)
    assert csv_frames.shape[1] == num_dofs + 7
    csv_pos = csv_frames[:, 0:3]
    csv_rot = csv_frames[:, 3:7]
    csv_q = csv_frames[:, 7:]
    csv_q = csv_q * csv_q_w + csv_q_b
    csv_q = csv_q[:, csv_dof_map]
    if q_min is not None:
        csv_q = np.clip(csv_q, q_min, q_max)
    rec = {
        'states_q': csv_q,
        'states_q_ctrl': csv_q,
        'states_quat': csv_rot,
        'states_pos': csv_pos,
        'args': {
            'dt': 0.033,
        }
    }
    return rec
