import numpy as np

REF_POINT_L = np.array([0.2443, 0.1517, -0.0455], dtype=float)  # waist to left hand
REF_POINT_M = np.array([0.2443, 0, -0.0455], dtype=float)  # waist to mid-point of two hands
REF_POINT_R = np.array([0.2443, -0.1517, -0.0455], dtype=float)  # waist to right hand

def planner_3dim(states_left_target, states_right_target, old_commands, old_reach_mask):

    if np.sum(states_left_target) < 0.001 or np.sum(states_right_target) < 0.001:
        commands = np.array([0.0, 0.0, 0.0])
        reach_mask = [0.0]
    else:
        ref_point = REF_POINT_M  # waist to mid-point of two hands
        tgt_point = (states_left_target[:3] + states_right_target[:3]) / 2

        commands = _calculate_commands(ref_point, tgt_point)
        # import ipdb; ipdb.set_trace()
        commands = _smooth_commands(old_commands, commands)
        reach_mask = _calculate_reach_mask(commands)
        reach_mask = _smooth_reach_mask(old_reach_mask, reach_mask)
    # print("planner_3dim: old_commands", old_commands, "old_reach_mask", old_reach_mask)
    # print("planner_3dim: commands", commands, "reach_mask", reach_mask, "states_left_target", states_left_target,)

    return {
        'commands': commands,
        'reach_mask': reach_mask,
    }

def _calculate_commands(ref_point, tgt_point):

    dx = tgt_point[0] - ref_point[0]
    dy = tgt_point[1] - ref_point[1]
    L = np.sqrt(dx**2 + dy**2)
    if -np.abs(dy) < 0.001 and np.abs(dx) < 0.001:
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
    old_val = float(old_mask[0]) if isinstance(old_mask, (list, np.ndarray)) else float(old_mask)
    new_val = float(new_mask[0]) if isinstance(new_mask, (list, np.ndarray)) else float(new_mask)
    if (new_val - old_val) < -step:
        return [old_val - step]
    elif (new_val - old_val) > step:
        return [old_val + step]
    else:
        return [new_val]

    # def cmd2out1step(self, scale=3):
    #     step = scale * self.dt

    #     up_mask = (self.tgt_commands-self.commands[:, :3])<-step
    #     down_mask = (self.tgt_commands-self.commands[:, :3])>step
    #     others = ~torch.logical_or(up_mask, down_mask)

    #     self.tgt_commands[up_mask] += step
    #     self.tgt_commands[down_mask] -= step
    #     self.tgt_commands[others] = self.commands[:, :3][others]

    # def mask2stage1step(self, scale=5):
    #     step = scale * self.dt

    #     _mask_float = self.standing_envs_mask.to(dtype=torch.float32)

    #     up_mask = (self.standing_stages-_mask_float)<-step
    #     down_mask = (self.standing_stages-_mask_float)>step
    #     others = ~torch.logical_or(up_mask, down_mask)

    #     self.standing_stages[up_mask] += step
    #     self.standing_stages[down_mask] -= step
    #     self.standing_stages[others] = _mask_float[others].to(dtype=torch.float32)

# class ThreeDimPlanner(BasePlanner):
#     def __init__(self, cfg: G1ReachCfg, device, num_envs):
#         super().__init__(cfg, device, num_envs, command_len=3)

#     def plan(self, target_hand_states, current_hand_states):
#         # under base coordinate system
#         dx, dy, L, yaw = self.get_relative_target(target_hand_states, current_hand_states)


#         self.commands[:, 0] = torch.clip(
#             self.planner_cfg.kp_lin_x * dx,
#             min=self.planner_cfg.lin_vel_x[0],
#             max=self.planner_cfg.lin_vel_x[1],
#         )
#         self.commands[:, 1] = torch.clip(
#             self.planner_cfg.kp_lin_y * dy,
#             min=self.planner_cfg.lin_vel_y[0],
#             max=self.planner_cfg.lin_vel_y[1],
#         )
#         self.commands[:, 2] = torch.clip(
#             self.planner_cfg.kp_ang_yaw * yaw, 
#             min=self.planner_cfg.ang_vel_yaw[0],
#             max=self.planner_cfg.ang_vel_yaw[1]
#         )

#         mask_L = (L < self.planner_cfg.L_threshold) & (torch.abs(yaw) < self.planner_cfg.yaw_threshold_when_close)

#         self.commands[mask_L, :] = 0.0

#         return self.commands


kp_lin_x = 1.0
kp_lin_y = 0.5
kp_ang_yaw = 1.0
lin_vel_x = [-0.3, 0.6]  # min max [m/s]
lin_vel_y = [-0.3, 0.3]  # min max [m/s]
ang_vel_yaw = [-0.6, 0.6]  # min max [rad/s]
L_threshold = 0.05
yaw_threshold_when_close = np.pi / 4
standing_lin_thrd = 0.1
standing_ang_thrd = 0.1
dt = 0.02