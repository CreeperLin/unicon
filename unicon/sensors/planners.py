import numpy as np

REF_POINT_L = np.array([0.2443, 0.1517, -0.0455], dtype=float)  # waist to left hand
REF_POINT_M = np.array([0.2443, 0, -0.0455], dtype=float)  # waist to mid-point of two hands
REF_POINT_R = np.array([0.2443, -0.1517, -0.0455], dtype=float)  # waist to right hand

# def planner_3dim(states_left_target, states_right_target, states_reach_mask, old_commands, old_standing_stage):

#     if np.sum(states_left_target) < 0.001 or np.sum(states_right_target) < 0.001:
#         commands = np.array([0.0, 0.0, 0.0])
#         reach_mask = [0.0]
#     else:
#         ref_point = REF_POINT_M  # waist to mid-point of two hands
#         tgt_point = (states_left_target[:3] + states_right_target[:3]) / 2

#         commands = _calculate_commands_3dim(ref_point, tgt_point)
#         # import ipdb; ipdb.set_trace()
#         commands = _smooth_commands(old_commands, commands)
#         reach_mask = _calculate_reach_mask(commands)
#     # print("planner_3dim: old_commands", old_commands, "old_reach_mask", old_reach_mask)
#     # print("planner_3dim: commands", commands, "reach_mask", reach_mask, "states_left_target", states_left_target,)

#     return {
#         'commands': commands,
#         'reach_mask': reach_mask,
#     }

# def planner_dummy(states_left_target, states_right_target, states_reach_mask, old_commands, old_standing_stage):

#     if np.sum(states_left_target) < 0.001 or np.sum(states_right_target) < 0.001:
#         commands = np.array([0.0, 0.0, 0.0])
#         reach_mask = [0.0, 0.0]
#         standing_stage = [0.0, 0.0]
#     else:
#         commands = np.array([0.0, 0.0, 0.0])
#         reach_mask = [1.0, 1.0]
#         standing_stage = [1.0, 1.0]

#     commands = _smooth_commands(old_commands, commands)
#     # reach_mask = _calculate_reach_mask(commands)
#     standing_stage = _smooth_reach_mask(old_standing_stage, standing_stage)

#     return {
#         'commands': commands,
#         'standing_stage': standing_stage,
#     }


def planner_fix(states_left_target, states_right_target, states_reach_mask, old_commands, old_standing_stage):
    
    reach_mask = [states_reach_mask[0], states_reach_mask[1]]

    if reach_mask[0] == 0.0 and reach_mask[1] == 0.0:
        commands = np.array([0.0, 0.0, 0.0])
    else:
        if reach_mask[0] == 1.0 and reach_mask[1] == 0.0:
            ref_point = REF_POINT_L
            tgt_point = states_left_target[:3]
        elif reach_mask[0] == 0.0 and reach_mask[1] == 1.0:
            ref_point = REF_POINT_R
            tgt_point = states_right_target[:3]
        else:
            ref_point = REF_POINT_M
            tgt_point = (states_left_target[:3] + states_right_target[:3]) / 2
    
        commands = _calculate_commands_fix(ref_point, tgt_point)

    commands = _smooth_commands(old_commands, commands)
    # reach_mask = _calculate_reach_mask(commands)
    standing_stage = _smooth_reach_mask(old_standing_stage, reach_mask)

    return {
        'commands': commands,
        'standing_stage': standing_stage,
    }


def _calculate_commands_3dim(ref_point, tgt_point):

    dx = tgt_point[0] - ref_point[0]
    dy = tgt_point[1] - ref_point[1]
    L = np.sqrt(dx**2 + dy**2)
    if np.abs(dy) < 0.001 and np.abs(dx) < 0.001:
        yaw = 0.0 # np.arctan2 do not has this case
    else:
        yaw = np.arctan2(dy, dx)

    lin_vel_x_cmd = np.clip(kp_lin_x * dx, lin_vel_x[0], lin_vel_x[1])
    lin_vel_y_cmd = np.clip(kp_lin_y * dy, lin_vel_y[0], lin_vel_y[1])
    ang_vel_yaw_cmd = np.clip(kp_ang_yaw * yaw, ang_vel_yaw[0], ang_vel_yaw[1])

    if (L < L_threshold) and (np.abs(yaw) < yaw_threshold_when_close):
        return np.array([0.0, 0.0, 0.0])
    else:
        return np.array([lin_vel_x_cmd, lin_vel_y_cmd, ang_vel_yaw_cmd])
    
def _calculate_commands_fix(ref_point, tgt_point):

    dx = tgt_point[0] - ref_point[0]
    dy = tgt_point[1] - ref_point[1]
    L = np.sqrt(dx**2 + dy**2)
    if np.abs(dy) < 0.1 and np.abs(dx) < 0.1:
        yaw = 0.0 # np.arctan2 do not has this case
    else:
        yaw = np.arctan2(dy, dx)

    if dx < -standing_lin_thrd:
        lin_vel_x_cmd = lin_vel_x[0]
    elif dx > standing_lin_thrd:
        lin_vel_x_cmd = lin_vel_x[1]
    else:
        lin_vel_x_cmd = 0.0

    if dy < -standing_lin_thrd:
        lin_vel_y_cmd = lin_vel_y[0]
    elif dy > standing_lin_thrd:
        lin_vel_y_cmd = lin_vel_y[1]
    else:
        lin_vel_y_cmd = 0.0

    if yaw < -standing_ang_thrd:
        ang_vel_yaw_cmd = ang_vel_yaw[0]
    elif yaw > standing_ang_thrd:
        ang_vel_yaw_cmd = ang_vel_yaw[1]
    else:
        ang_vel_yaw_cmd = 0.0

    return np.array([lin_vel_x_cmd, lin_vel_y_cmd, ang_vel_yaw_cmd])

def _calculate_reach_mask(commands):
        # _standing = torch.logical_and(
        #     torch.norm(self.commands[env_ids, :2], dim=1) <= self.cfg.commands.standing_lin_thrd if hasattr(self.cfg.commands, "standing_lin_thrd") else 0.2,
        #     torch.abs(self.commands[env_ids, 2]) <= self.cfg.commands.standing_ang_thrd if hasattr(self.cfg.commands, "standing_ang_thrd") else 0.15
        # )
    if np.linalg.norm(commands[:2]) <= standing_lin_thrd and np.abs(commands[2]) <= standing_ang_thrd:
        return [1.0]
    else:
        return [0.0]

def _smooth_commands(old_commands, new_commands, scale=3):
    step = scale * dt
    smoothed_commands = np.zeros(3)
    for i in range(len(old_commands)):
        delta = new_commands[i] - old_commands[i]
        if delta > step:
            smoothed_commands[i] = old_commands[i] + step
        elif delta < -step:
            smoothed_commands[i] = old_commands[i] - step
        else:
            smoothed_commands[i] = new_commands[i]
    return smoothed_commands

def _smooth_reach_mask(old_mask, new_mask, scale=5):
    step = scale * dt
    smoothed_mask = np.zeros(2)
    for i in range(len(old_mask)):
        old_val = old_mask[i]
        new_val = new_mask[i]
        if (new_val - old_val) < -step:
            smoothed_mask[i] = old_val - step
        elif (new_val - old_val) > step:
            smoothed_mask[i] = old_val + step
        else:
            smoothed_mask[i] = new_val
    return smoothed_mask


kp_lin_x = 1.0
kp_lin_y = 0.5
kp_ang_yaw = 1.0
lin_vel_x = [-0.3, 0.6]  # min max [m/s]
lin_vel_y = [-0.3, 0.3]  # min max [m/s]
ang_vel_yaw = [-0.6, 0.6]  # min max [rad/s]
L_threshold = 0.05
yaw_threshold_when_close = np.pi / 4
standing_lin_thrd = 0.1
standing_ang_thrd = np.pi / 2
dt = 0.02   