import yaml
import numpy as np


def load(yaml_path, dof_names_seq=None):
    from unicon.utils import get_ctx, expect
    ctx = get_ctx()

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    states_pos = None
    states_quat = None
    states_q = None

    components = data.get("components", [])
    joint_comp = None
    for comp in components:
        if comp.get("type") == "MultiValueSeq" and comp.get("content") == "JointDisplacement":
            joint_comp = comp
            break

    if joint_comp is not None:
        frames = joint_comp["frames"]
        frame_rate = joint_comp["frame_rate"]
        num_parts = joint_comp["num_parts"]

        q = np.array(frames, dtype=np.float32)

        robot_def = ctx['robot_def']
        dof_names = robot_def["DOF_NAMES"]
        q_min = robot_def.get("Q_CTRL_MIN")
        q_max = robot_def.get("Q_CTRL_MAX")

        dof_names_seq = dof_names if dof_names_seq is None else dof_names_seq
        expect(q.shape[1] == num_parts == len(dof_names_seq))

        if q_min is not None:
            q = np.clip(q, q_min, q_max)
        states_q = q

    dt = 1.0 / frame_rate

    rec = {
        "states_q": states_q,
        "states_q_ctrl": states_q,
        "states_pos": states_pos,
        "states_quat": states_quat,
        "args": {
            "dt": dt,
        }
    }

    return rec
