import struct
import json
import numpy as np


def servo_on_fsa(fi_fsa, fsa_ips):
    s = fi_fsa.s
    fsa_port_fast = fi_fsa.fsa_port_fast

    def fast_set_position_mode(server_ip):
        tx_messages = struct.pack('>B', 0x04)
        s.sendto(tx_messages, (server_ip, fsa_port_fast))

    def fast_set_position_control_2(server_ip, position):
        msg = struct.pack('>Bfxxxxxxxx', 0x0A, position)
        s.sendto(msg, (server_ip, fsa_port_fast))

    def fast_set_enable(server_ip):
        tx_messages = struct.pack('>B', 0x01)
        s.sendto(tx_messages, (server_ip, fsa_port_fast))

    for ip in fsa_ips:
        fast_set_position_control_2(ip, 0.)
    for ip in fsa_ips:
        fast_set_enable(ip)
    for ip in fsa_ips:
        fast_set_position_mode(ip)


def servo_off_fsa(fi_fsa, fsa_ips):
    s = fi_fsa.s
    fsa_port_fast = fi_fsa.fsa_port_fast

    def fast_set_disable(server_ip):
        tx_messages = struct.pack('>B', 0x02)
        s.sendto(tx_messages, (server_ip, fsa_port_fast))

    for ip in fsa_ips:
        fast_set_disable(ip)


def fsa_broadcast(fi_fsa, filter_type=None, max_ips=None, max_timeouts=10):
    s = fi_fsa.s
    fsa_network = fi_fsa.fsa_network
    fsa_port_comm = fi_fsa.fsa_port_comm
    address_list = []
    s.sendto("Is any fourier smart server here?".encode("utf-8"), (fsa_network, fsa_port_comm))
    n_to = 0
    while True:
        try:
            data, address = s.recvfrom(1024)
            ip = address[0]
            if filter_type is None:
                address_list.append(ip)
                continue
            json_obj = json.loads(data.decode("utf-8"))
            if "type" in json_obj:
                if json_obj["type"] == filter_type:
                    address_list.append(ip)
        except TimeoutError:
            n_to += 1
            if n_to >= max_timeouts:
                break
            if max_ips is not None and len(address_list) >= max_ips:
                break
    return address_list


def reboot_fsa(fi_fsa, server_ip):
    s = fi_fsa.s
    fsa_port_ctrl = fi_fsa.fsa_port_ctrl
    data = {"method": "SET", "reqTarget": "/reboot", "property": ""}
    json_str = json.dumps(data)
    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))


def set_control_param_imm(fi_fsa, server_ip, dct):
    s = fi_fsa.s
    fsa_port_ctrl = fi_fsa.fsa_port_ctrl
    data = {
        "method": "SET",
        "reqTarget": "/control_param_imm",
        "property": "",
        "motor_max_speed_imm": dct["motor_max_speed"],
        "motor_max_acceleration_imm": dct["motor_max_acceleration"],
        "motor_max_current_imm": dct["motor_max_current"],
    }
    json_str = json.dumps(data)
    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))


def get_control_param_imm(fi_fsa, server_ip):
    s = fi_fsa.s
    fsa_port_ctrl = fi_fsa.fsa_port_ctrl
    data = {
        "method": "GET",
        "reqTarget": "/control_param_imm",
        "property": "",
    }
    json_str = json.dumps(data)
    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        while True:
            try:
                data, address = s.recvfrom(1024)
            except TimeoutError:
                print('timeout', server_ip)
                return None
            if address[0] != server_ip:
                continue
            break
        json_obj = json.loads(data.decode("utf-8"))
        return json_obj
    except Exception:
        return None


def set_control_param(fi_fsa, server_ip, dct):
    s = fi_fsa.s
    fsa_port_ctrl = fi_fsa.fsa_port_ctrl
    data = {
        "method": "SET",
        "reqTarget": "/control_param",
        "property": "",
        "motor_max_speed": dct["motor_max_speed"],
        "motor_max_acceleration": dct["motor_max_acceleration"],
        "motor_max_current": dct["motor_max_current"],
    }
    json_str = json.dumps(data)
    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))


def get_root_infos(fi_fsa, fsa_ips):
    s = fi_fsa.s
    fsa_port_ctrl = fi_fsa.fsa_port_ctrl

    def get_root(server_ip):
        data = {"method": "GET", "reqTarget": "/", "property": ""}

        json_str = json.dumps(data)
        s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
        while True:
            try:
                data, address = s.recvfrom(1024)
            except TimeoutError:
                print('timeout', server_ip)
                return None
            if address[0] != server_ip:
                continue
            break
        json_obj = json.loads(data.decode("utf-8"))
        return json_obj

    from collections import defaultdict
    dct = defaultdict(list)
    for ip in fsa_ips:
        j = get_root(ip)
        if j is None:
            continue
        for k, v in j.items():
            dct[k].append(v)
    return dct


def get_comm_infos(fi_fsa, fsa_ips):
    s = fi_fsa.s
    fsa_port_comm = fi_fsa.fsa_port_comm

    def get_comm_root(server_ip):
        data = {"method": "GET", "reqTarget": "/", "property": ""}

        json_str = json.dumps(data)
        s.sendto(str.encode(json_str), (server_ip, fsa_port_comm))
        while True:
            data, address = s.recvfrom(1024)
            if address[0] != server_ip:
                continue
            break
        json_obj = json.loads(data.decode("utf-8"))
        return json_obj

    from collections import defaultdict
    dct = defaultdict(list)
    for ip in fsa_ips:
        j = get_comm_root(ip)
        for k, v in j.items():
            dct[k].append(v)
    return dct


def battery_measure(server_ip="192.168.137.202"):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(1)  # set timeout to 1 second
    data = {
        "method": "GET",
        "reqTarget": "/battery_voltage",
        "property": "",
    }
    json_str = json.dumps(data)
    s.sendto(str.encode(json_str), (server_ip, 2334))
    try:
        data, address = s.recvfrom(1024)
        json_obj = json.loads(data.decode('utf-8'))
        return json_obj["battery"]
    except socket.timeout:  # fail after 1 second of no activity
        print("Didn't receive anymore data! [Timeout]")
        return None


def get_battery_level():
    for ip in ["192.168.137.202", "192.168.137.212"]:
        ret = battery_measure(ip)
        if ret is None:
            continue
        return ret


_default_fsa_ips = [
    # left leg
    "192.168.137.70",
    "192.168.137.71",
    "192.168.137.72",
    "192.168.137.73",
    "192.168.137.74",
    "192.168.137.75",
    # right leg
    "192.168.137.50",
    "192.168.137.51",
    "192.168.137.52",
    "192.168.137.53",
    "192.168.137.54",
    "192.168.137.55",
    # waist
    "192.168.137.90",
    "192.168.137.91",
    "192.168.137.92",
    # head
    "192.168.137.93",
    "192.168.137.94",
    "192.168.137.95",
    # left arm
    "192.168.137.10",
    "192.168.137.11",
    "192.168.137.12",
    "192.168.137.13",
    "192.168.137.14",
    "192.168.137.15",
    "192.168.137.16",
    # right arm
    "192.168.137.30",
    "192.168.137.31",
    "192.168.137.32",
    "192.168.137.33",
    "192.168.137.34",
    "192.168.137.35",
    "192.168.137.36",
]


def pcap2rec(path, dt=0.02, fsa_ips=None):
    fsa_ips = _default_fsa_ips if fsa_ips is None else fsa_ips
    # host_ip = '192.168.137.254'
    ip2idx = {ip: i for i, ip in enumerate(fsa_ips)}
    num_dofs = len(ip2idx)
    import dpkt
    from dpkt.utils import inet_to_str
    send_tss = [[] for _ in range(len(fsa_ips))]
    pos_controls = [[] for _ in range(len(fsa_ips))]
    tss = [[] for _ in range(len(fsa_ips))]
    positions = [[] for _ in range(len(fsa_ips))]
    velocities = [[] for _ in range(len(fsa_ips))]
    currents = [[] for _ in range(len(fsa_ips))]
    for ts, buf in dpkt.pcap.Reader(open(path, 'rb')):
        eth = dpkt.ethernet.Ethernet(buf)
        if not isinstance(eth.data, dpkt.ip.IP):
            print('Non IP Packet type not supported %s\n' % eth.data.__class__.__name__)
            continue
        ip = eth.data
        if not isinstance(ip.data, dpkt.udp.UDP):
            continue
        udp = ip.data
        pkt = udp.data
        if udp.sport == 2335:
            ip_src = inet_to_str(ip.src)
            src_idx = ip2idx.get(ip_src)
            b, pos, vel, cur = struct.unpack('>Bfff', pkt)
            assert b == 26
            tss[src_idx].append(ts)
            positions[src_idx].append(pos)
            velocities[src_idx].append(vel)
            currents[src_idx].append(cur)
        elif udp.dport == 2335 and pkt[0] == 0x0A:
            ip_dst = inet_to_str(ip.dst)
            dst_idx = ip2idx.get(ip_dst)
            b, pos_c, vel_ff, cur_ff = struct.unpack('>Bfff', pkt)
            send_tss[dst_idx].append(ts)
            pos_controls[dst_idx].append(pos_c)
    print([len(v) for v in tss])
    vs = [
        tss,
        positions,
        velocities,
        currents,
        send_tss,
        pos_controls,
    ]

    def proc(x):
        max_len = max([len(xx) for xx in x])
        nx = []
        ix = 0
        for xx in x:
            if len(xx) != 0:
                ix = xx[0]
            z = [ix] * (max_len - len(xx)) + xx
            nx.append(z)
        return np.array(nx)

    vs = [proc(v) for v in vs]
    tss, positions, velocities, currents, send_tss, pos_controls = vs
    pos_controls = np.deg2rad(pos_controls)
    positions = np.deg2rad(positions)
    velocities = np.deg2rad(velocities)
    pos_controls = pos_controls if len(pos_controls[0]) else positions.copy()
    send_tss = send_tss if len(send_tss[0]) else tss.copy()
    print(tss.shape)
    print(positions.shape)
    print(velocities.shape)
    print(currents.shape)
    print(send_tss.shape)
    print(pos_controls.shape)
    t0 = min(np.min(tss[:, 0]), np.min(send_tss[:, 0]))
    t1 = max(np.max(tss[:, -1]), np.max(send_tss[:, -1]))
    dura = t1 - t0
    num_frames = int(dura / dt)
    print('t0', t0, 't1', t1)
    print('dura', dura)
    print('num_frames', num_frames)

    states_q_ctrl = np.zeros((num_frames, num_dofs))
    states_q = np.zeros((num_frames, num_dofs))
    states_qd = np.zeros((num_frames, num_dofs))
    states_q_cur = np.zeros((num_frames, num_dofs))

    send_tss -= t0
    tss -= t0
    t0 = 0
    # print('send_tss', send_tss[0])
    import torch
    _tss = torch.from_numpy(tss)
    _send_tss = torch.from_numpy(send_tss)

    st = torch.zeros(num_dofs, 1)
    b_inds = np.arange(num_dofs)
    for i in range(num_frames):
        ct = t0 + i * dt
        st[:] = ct
        send_inds = torch.searchsorted(_send_tss, st, side='left').squeeze().numpy()
        # print(i, ct, send_inds)
        send_inds[send_inds >= pos_controls.shape[1]] = 0
        states_q_ctrl[i] = pos_controls[b_inds, send_inds]
        recv_inds = torch.searchsorted(_tss, st, side='left').squeeze().numpy()
        recv_inds[recv_inds >= positions.shape[1]] = 0
        states_q[i] = positions[b_inds, recv_inds]
        states_qd[i] = velocities[b_inds, recv_inds]
        states_q_cur[i] = currents[b_inds, recv_inds]

    rec = {
        'states_q': states_q,
        'states_qd': states_qd,
        'states_q_cur': states_q_cur,
        'states_q_ctrl': states_q_ctrl,
        'states_q_ctrl': states_q_ctrl,
        # 'states_quat': None,
        'send_ts': send_tss.T,
        'recv_ts': tss.T,
        'args': {},
    }
    return rec


def gains_pd2pv(
    pd_kp,
    pd_kd,
    pole_pairs,
    reduction_ratio,
    kt,
):
    G_w = pole_pairs * reduction_ratio / 360
    G_theta = reduction_ratio
    G_pd = np.pi / 180

    position_kp = (pd_kp * G_w) / (pd_kd * G_theta)
    velocity_kp = (pd_kd * G_pd) / (reduction_ratio * kt * G_w)

    return position_kp, velocity_kp


def gains_pv2pd(
    position_kp,
    velocity_kp,
    pole_pairs,
    reduction_ratio,
    kt,
):
    G_w = pole_pairs * reduction_ratio / 360
    G_theta = reduction_ratio
    G_pd = np.pi / 180

    pd_kp = (position_kp * velocity_kp * reduction_ratio * kt * G_theta) / G_pd
    pd_kd = (reduction_ratio * kt * velocity_kp * G_w) / G_pd

    return pd_kp, pd_kd


if __name__ == '__main__':
    print('bat', get_battery_level())
