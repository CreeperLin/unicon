def cb_system_ut_dex3(
    states_hand_q=None,
    states_hand_q_ctrl=None,
    states_hand_tactile=None,
    iface='eth0',
    cmd_topic='rt/dex3/*/cmd',
    # dex3_state_topic='rt/lf/dex3/*/state',
    state_topic='rt/dex3/*/state',
    async_recv=False,
):
    import time
    from unicon.utils import get_ctx, list2slice, expect
    ctx = get_ctx()
    hand_def = ctx['robot_def']['hand_def']
    NAME = hand_def.get('NAME')
    expect('dex3' in NAME)
    DOF_NAMES = hand_def['DOF_NAMES']

    def pack_mode(mid, status=0x01, timeout=0x00) -> int:
        mode = 0
        mode |= (mid & 0x0F)
        mode |= (status & 0x07) << 4
        mode |= (timeout & 0x01) << 7
        return mode

    from unicon.utils.unitree import init_channel
    from unitree_sdk2py.core.channel import (
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
    cmd_type = HandCmd_
    state_type = HandState_
    cmd_def_cls = unitree_hg_msg_dds__HandCmd_
    cmd = cmd_def_cls()
    for i, c in enumerate(cmd.motor_cmd):
        mode = pack_mode(mid=i, status=0x01, timeout=0x00)
        c.mode = mode
        c.q = 0.
        c.dq = 0.
        c.tau = 0.
        c.kp = 0.5
        c.kd = 0.1

    domain_id = 0

    init_channel(
        domain_id,
        networkInterface=iface,
    )
    sides = ['left', 'right']
    joint_pub_specs = []
    joint_sub_specs = []
    for side in sides:
        pub = ChannelPublisher(cmd_topic.replace('*', side), cmd_type)
        pub.Init()
        sub = ChannelSubscriber(state_topic.replace('*', side), state_type)
        sub.Init(None, 0)

        dof_names_side = [n for n in DOF_NAMES if side in n]
        num_dofs_side = len(dof_names_side)
        dof_inds_side = [DOF_NAMES.index(n) for n in dof_names_side]
        dof_inds_sl_side = list2slice(dof_inds_side)
        print('num_dofs_side', num_dofs_side)
        print('dof_inds_sl_side', dof_inds_sl_side)
        joint_pub_specs.append((pub, cmd, dof_inds_sl_side))
        joint_sub_specs.append((sub, dof_inds_sl_side))

    for i in range(10):
        print('waiting for sub', i)
        state = sub.Read()
        print(state)
        if len(state.motor_state):
            break
        time.sleep(1)
    else:
        raise RuntimeError('sub read timeout')

    def cb_recv():
        for sub, dof_inds_sl in joint_sub_specs:
            state = sub.Read()
            if state is None:
                return True
            motors = state.motor_state
            q = [m.q for m in motors]
            states_hand_q[dof_inds_sl] = q
            # dq = [m.velocity for m in motors]
            # states_qd[dof_inds_sl] = dq

    def cb_send():
        q_ctrl = states_hand_q_ctrl
        for pub, cmd, dof_inds_sl in joint_pub_specs:
            for c, qc in zip(cmd.motor_cmd, q_ctrl[dof_inds_sl]):
                c.q = qc
            pub.Write(cmd)

    def cb_close():
        for pub, cmd, _ in joint_pub_specs:
            for c in cmd.motor_cmd:
                mode = pack_mode(mid=i, status=0x01, timeout=0x01)
                c.mode = mode
                c.q = 0.
                c.dq = 0.
                c.tau = 0.
                c.kp = 0.
                c.kd = 0.
            pub.Write(cmd)
            pub.Close()

    return cb_recv, cb_send, cb_close
