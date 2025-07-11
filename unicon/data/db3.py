import struct
from collections import namedtuple

_spec_header = {
    'seq': 'uint32',
    'secs': 'uint32',
    'nsecs': 'uint32',
    'frame_id': 'char[]',
}

_spec_pose2d = {
    'x': 'float64',
    'y': 'float64',
    'theta': 'float64',
}

_spec_quat = {
    'x': 'float64',
    'y': 'float64',
    'z': 'float64',
    'w': 'float64',
}

_spec_point = {
    'x': 'float64',
    'y': 'float64',
    'z': 'float64',
}

_spec_pose = {
    'position': 'geometry_msgs/Point',
    'orientation': 'geometry_msgs/Quaternion',
}

_msg_specs = {
    'std_msgs/Header': _spec_header,
    'geometry_msgs/Pose2D': _spec_pose2d,
    'geometry_msgs/Pose': _spec_pose,
    'geometry_msgs/Quaternion': _spec_quat,
    'geometry_msgs/Point': _spec_point,
}

_tuples = {}
_dtype2fmt = {
    'uint32': 'I',
    'float64': 'd',
    'char': 's',
}


def parse_msg2(msg_type, data, ofs=0):
    spec = _msg_specs.get(msg_type, msg_type)
    if isinstance(spec, str):
        spec = [spec]
    if isinstance(spec, (tuple, list)):
        dtype = spec[0]
        num = spec[1] if len(spec) > 1 else 1
        is_array = False
        if dtype.endswith('[]'):
            is_array = True
            dtype = dtype[:-2]
        elm_fmt = _dtype2fmt[dtype]
        if is_array:
            if dtype == 'char':
                sz_fmt = 'I'
                sz = 4
            else:
                sz_fmt = 'Q'
                sz = 8
            arr_len = struct.unpack_from(f'<{sz_fmt}', data, ofs)[0]
            assert arr_len < 1024
            ofs = ofs + sz
            fmt = f'<{arr_len}{elm_fmt}'
        else:
            fmt = f'<{num}{elm_fmt}'
        # print(msg_type, ofs, fmt)
        sz = struct.calcsize(fmt)
        msg = struct.unpack_from(fmt, data, ofs)
        msg = msg if is_array else msg[0]
        msg = msg[0].decode() if spec[0] == 'char[]' else msg
        ofs += sz
        # print(msg, sz, ofs)
        return msg, ofs
    # print(msg_type, spec, ofs)
    assert isinstance(spec, dict)
    fields = {}
    for k, v in spec.items():
        m, ofs = parse_msg2(v, data, ofs)
        fields[k] = m
    type_name = msg_type.split('/')[-1]
    msg_cls = _tuples.get(msg_type)
    if msg_cls is None:
        msg_cls = namedtuple(type_name, fields.keys())
        _tuples[msg_type] = msg_cls
    msg = msg_cls(*fields.values())
    return msg, ofs


def load(
    path=None,
    msg_specs=None,
    topic_names=None,
    states_map=None,
    dof_names=None,
    dt=None,
):
    from unicon.utils import get_ctx
    robot_def = get_ctx()['robot_def']
    robot_dof_names = robot_def.get('DOF_NAMES')
    if msg_specs is not None:
        _msg_specs.update(msg_specs)
    import sqlite3
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute('SELECT id, name, type FROM topics')
    topic_rows = cursor.fetchall()
    topic_type_map = {row[1]: row[2] for row in topic_rows}
    topic_id_map = {row[0]: row[1] for row in topic_rows}
    print('topic_type_map', topic_type_map)
    print('topic_id_map', topic_id_map)

    cursor.execute('SELECT topic_id, timestamp, data FROM messages')
    message_rows = cursor.fetchall()
    topic_names = [topic_names] if isinstance(topic_names, str) else topic_names

    msgs = {k: [] for k in topic_names}
    for topic_id, timestamp, data in message_rows:
        _topic_name = topic_id_map[topic_id]
        if _topic_name not in topic_names:
            continue
        msg_type_str = topic_type_map[_topic_name]
        msg, _ = parse_msg2(msg_type_str, data)
        msgs[_topic_name].append(msg)

    conn.close()

    def follow(obj, path):
        keys = path.split('.')
        for k in keys:
            obj = getattr(obj, k)
        return obj

    import numpy as np
    states = {}
    for k, _msgs in msgs.items():
        ts = [m.header.secs + m.header.nsecs * 1e-9 for m in _msgs]
        ts = np.array(ts)
        s_map = states_map.get(k)
        for p, sk in s_map.items():
            s = [follow(m, p) for m in _msgs]
            states[sk] = np.array(s)
            states[sk + '.ts'] = ts

    if dof_names is not None:
        num_dofs = len(dof_names)
        dof_map = [robot_dof_names.index(n) for n in dof_names]
        for k, v in states.items():
            if len(v.shape) < 2 or v.shape[1] != num_dofs:
                continue
            ns = np.zeros((len(v), len(robot_dof_names)))
            ns[:, dof_map] = v
            states[k] = ns
    if dt is not None:
        from unicon.utils import states_ts2dt
        states = states_ts2dt(states, dt=dt)
    print({k: v.shape for k, v in states.items()})

    rec = {
        'dt': dt,
    }
    rec.update(states)
    return rec
