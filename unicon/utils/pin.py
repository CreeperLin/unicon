import os
import numpy as np
try:
    import pinocchio as pin
except ImportError:
    pin = None


def load_robot_pin(
    urdf_path=None,
    mjcf_path=None,
):
    robot_pin = pin.RobotWrapper.BuildFromURDF(
        # model = pin.buildModelFromUrdf(
        filename=urdf_path,
        package_dirs=[os.path.dirname(urdf_path)],
        root_joint=None,
    )
    return robot_pin


def mat2se3(matrix):
    if matrix.shape != (4, 4):
        raise ValueError("Expected (4,4) homogeneous transform")
    return pin.SE3(matrix[:3, :3], matrix[:3, 3])


def dedup_frames(model):
    model = model.copy()
    from collections import defaultdict
    frame_ids = defaultdict(list)
    frames = model.frames
    for i, f in enumerate(frames):
        frame_ids[f.name].append(i)
    dups = {k: v for k, v in frame_ids.items() if len(v) > 1}
    if not len(dups):
        return model
    print('dedup_frames', dups)
    for k, v in dups.items():
        for i in range(1, len(v)):
            frames[v[i]].name = f'{k}_{i}'
    return model


def get_joint_names(robot_pin):
    model, data = robot_pin.model, robot_pin.data
    nq = model.nq
    names = [None for _ in range(nq + 1)]
    for f in model.frames:
        n = f.name
        idx = model.getJointId(n)
        if 0 < idx <= nq:
            names[idx] = n
    return names[1:]


def pin_fk(q, robot_pin=None, dof_names=None, link_names=None):
    model, data = robot_pin.model, robot_pin.data
    if link_names is None:
        x_inds_pin = range(len(model.frames))
    else:
        frame_names = [f.name for f in model.frames]
        x_inds_pin = [frame_names.index(n) for n in link_names]
    #     print('frame_names', frame_names)
    # print('x_inds_pin', x_inds_pin)
    q_pin = q
    if dof_names is not None:
        q_pin = np.zeros(model.nq + 1, dtype=q.dtype)
        inds = list(map(model.getJointId, dof_names))
        q_pin[inds] = q
        q_pin = q_pin[1:]
    pin.forwardKinematics(model, data, q_pin)
    pin.updateFramePlacements(model, data)
    return np.stack([data.oMf[i].homogeneous for i in x_inds_pin], axis=0)
