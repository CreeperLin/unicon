# yapf: disable

hmn = {
    'URDF': 'humanoid/humanoid_1.urdf',
# URDF = 'humanoid/humanoid_lower.urdf'
}

adam = {
    'URDF': 'adam/adam_lite/urdf/adam_lite_1_1.urdf',
    'DOF_NAMES': [
        'hipPitch_Left',
        'hipRoll_Left',
        'hipYaw_Left',
        'kneePitch_Left',
        'anklePitch_Left',
        'ankleRoll_Left',
        'hipPitch_Right',
        'hipRoll_Right',
        'hipYaw_Right',
        'kneePitch_Right',
        'anklePitch_Right',
        'ankleRoll_Right',
        'waistRoll',
        'waistPitch',
        'waistYaw',
        'shoulderPitch_Left',
        'shoulderRoll_Left',
        'shoulderYaw_Left',
        'elbow_Left',
        'wristYaw_Left',
        'shoulderPitch_Right',
        'shoulderRoll_Right',
        'shoulderYaw_Right',
        'elbow_Right',
        'wristYaw_Right',
    ],
    'KP': [
        305.0, 700.0, 405.0, 305.0, 20.0, 10.0,
        305.0, 700.0, 405.0, 305.0, 20.0, 10.0,
        # 305.0, 700.0, 405.0, 305.0, 30.0, 0.0,
        # 305.0, 700.0, 405.0, 305.0, 30.0, 0.0,
        405.0, 405.0, 205.0,
        18.0, 9.0, 9.0, 9.0, 7.0,
        18.0, 9.0, 9.0, 9.0, 7.0,
    ],
    'KD': [
        6.1, 30.0, 6.1, 6.1, 1.25, 1.0,
        6.1, 30.0, 6.1, 6.1, 1.25, 1.0,
        # 6.1, 30.0, 6.1, 6.1, 2.25, 0.25,
        # 6.1, 30.0, 6.1, 6.1, 2.25, 0.25,
        6.1, 6.1, 4.1,
        1.0, 1.0, 1.0, 1.0, 0.5,
        1.0, 1.0, 1.0, 1.0, 0.5,
    ],
}

t1 = {
    'URDF': 'Booster-T1/T1_serial_simple_version.urdf',
    'KP': {
        'AAHead_yaw': 50,
        'Head_pitch': 50,
        'Waist': 100,
        # left-lower
        'Ankle_Pitch': 40,
        'Ankle_Roll': 30,
        'Hip_Pitch': 85,
        'Hip_Roll': 55,
        'Hip_Yaw': 55,
        'Knee_Pitch': 110,
        # left-upper
        'Elbow_Pitch': 33.75,
        'Elbow_Yaw': 33.75,
        'Shoulder_Pitch': 33.75,
        'Shoulder_Roll': 33.75,
    },
    'KD': {
        'AAHead_yaw': 0.5,
        'Head_pitch': 0.5,
        'Waist': 1.0,
        # left-lower
        'Ankle_Pitch': 0.5,
        'Ankle_Roll': 0.4,
        'Hip_Pitch': 1.25,
        'Hip_Roll': 0.75,
        'Hip_Yaw': 0.75,
        'Knee_Pitch': 1.5,
        # left-upper
        'Elbow_Pitch': 0.45,
        'Elbow_Yaw': 0.45,
        'Shoulder_Pitch': 0.45,
        'Shoulder_Roll': 0.45,
    },
    'Q_RESET': {
        'AAHead_yaw': 0,
        'Head_pitch': 0,
        'Waist': 0,
        # left-lower
        'Left_Ankle_Pitch': -0.29,
        'Left_Ankle_Roll': 0,
        'Left_Hip_Pitch': -0.25,
        'Left_Hip_Roll': 0,
        'Left_Hip_Yaw': 0,
        'Left_Knee_Pitch': 0.55,
        # left-upper
        'Left_Elbow_Pitch': 0,
        # 'Left_Elbow_Yaw': -1.6,
        'Left_Elbow_Yaw': 0.,
        'Left_Shoulder_Pitch': 0.5,
        'Left_Shoulder_Roll': -1.4,
        # right-lower
        'Right_Ankle_Pitch': -0.29,
        'Right_Ankle_Roll': 0,
        'Right_Hip_Pitch': -0.25,
        'Right_Hip_Roll': 0,
        'Right_Hip_Yaw': 0,
        'Right_Knee_Pitch': 0.55,
        # right-upper
        'Right_Elbow_Pitch': 0,
        # 'Right_Elbow_Yaw': 1.6,
        'Right_Elbow_Yaw': 0.,
        'Right_Shoulder_Pitch': 0.5,
        'Right_Shoulder_Roll': 1.4,
    },
}

dobot = {
    'URDF': 'dobot/urdf/dobot_simplified_hand.urdf',
}

pm = {
    'URDF': 'pm/urdf/serial_pm_v2.urdf',
}

ti5 = {
    'URDF': 'Appendix 2 - t1/urdf/t1_modify.urdf',
}
