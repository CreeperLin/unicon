def cb_viz_swv(
    states_q,
    states_quat=None,
    states_rpy=None,
    states_pos=None,
    urdf_path=None,
    host='localhost',
    port=6000,
    dof_names=None,
    start_server=True,
    init_z=1.0,
    use_rpy=False,
):
    import numpy as np
    from sim_web_visualizer import MeshCatVisualizerBase
    from sim_web_visualizer.parser.yourdfpy import URDF
    from unicon.utils import quat2mat_np, rpy2mat_np, try_conn

    if use_rpy:
        states_quat = None
    else:
        states_rpy = None

    port = int(port)
    proc = None
    if start_server and try_conn(host, port):
        print('starting meshcat-server')
        import time
        import subprocess
        args = ['meshcat-server']
        proc = subprocess.Popen(args)
        time.sleep(2)
        ret = proc.poll()
        if ret is not None:
            proc.kill()
            raise RuntimeError('meshcat-server failed')

    viz = MeshCatVisualizerBase(port=port, host=host)

    num_states_dofs = len(states_q)
    robot = URDF.load(str(urdf_path), build_tree=True)
    num_dofs = robot.num_dofs
    urdf_dof_names = robot.actuated_joint_names
    print('urdf_dof_names', len(urdf_dof_names), urdf_dof_names)
    print('dof_names', len(dof_names), dof_names)
    print(num_states_dofs, num_dofs)

    dof_map = None
    if dof_names is not None:
        dof_map = [dof_names.index(n) for n in urdf_dof_names]
    print('viz dof_map', dof_map)

    viz.viz['/URDF'].delete()
    asset_resource = viz.dry_load_asset(str(urdf_path), collapse_fixed_joints=False)
    viz.load_asset_resources(asset_resource, '/URDF', scale=1.0)
    urdf_viz = viz.viz['/URDF']
    # print(dir(urdf_viz))
    tf = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., init_z], [0., 0., 0., 1.]])
    urdf_viz.set_transform(tf)
    qpos = np.zeros(num_dofs)
    last_h = None
    pt = 0

    class cb_cls:

        def __call__(self):
            nonlocal last_h, pt
            h = (states_q + 100).sum()
            if last_h == h:
                return
            last_h = h
            if pt % 10 == 0:
                print('states_q', pt)
            pt += 1

            qpos[:] = states_q[dof_map]
            robot.update_kinematics(qpos)
            for link_name, link in robot.link_map.items():
                pose = robot.get_link_global_transform(link_name)
                urdf_viz[link_name].set_transform(pose)
            if states_quat is not None:
                tf[:3, :3] = quat2mat_np(states_quat)
            if states_rpy is not None:
                tf[:3, :3] = rpy2mat_np(states_rpy)
            if states_pos is not None:
                tf[:3, 3] = states_pos
            if states_pos is not None or states_quat is not None or states_rpy is not None:
                urdf_viz.set_transform(tf)

        def __del__(self):
            if proc is not None:
                print('killing meshcat-server')
                proc.kill()

    return cb_cls()


def run_viz(viz_type='swv', robot_type=None, **kwds):
    import os
    from unicon.utils import import_obj
    _default_urdf_root = f'{os.environ["HOME"]}/GitRepo/GR1/resources/robots/'
    _default_urdf_root = os.environ.get('UNICON_URDF_ROOT', _default_urdf_root)
    robot_def = import_obj((robot_type, None), default_mod_prefix='unicon.defs')
    DOF_NAMES = getattr(robot_def, 'DOF_NAMES', None)
    urdf_path = getattr(robot_def, 'URDF')
    urdf_path = os.path.join(_default_urdf_root, urdf_path)
    assert os.path.exists(urdf_path)
    from unicon.states import states_get
    from unicon.states import states_init
    states_init(use_shm=True, load=True, reuse=True)
    states_props = {
        'states_rpy': states_get('rpy'),
        'states_ang_vel': states_get('ang_vel'),
        'states_quat': states_get('quat'),
        'states_q': states_get('q'),
        'states_qd': states_get('qd'),
    }
    states_q = states_props['states_q']
    states_quat = states_props['states_quat']
    states_rpy = states_props['states_rpy']
    states_pos = states_get('pos')
    cb_viz_cls = import_obj(viz_type, default_name_prefix='cb_viz', default_mod_prefix='unicon.viz')
    dof_names = DOF_NAMES
    # dof_names = DOF_NAMES_2
    cb_viz = cb_viz_cls(
        states_q=states_q,
        states_quat=states_quat,
        states_rpy=states_rpy,
        states_pos=states_pos,
        dof_names=dof_names,
        urdf_path=urdf_path,
        **kwds,
    )
    from unicon.utils import loop_timed
    dt = 0.02
    loop_timed(cb_viz, dt=dt, sleep_fn='sleep_block')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-rt', '--robot_type', default='gr1t2')
    parser.add_argument('-vt', '--viz_type', default='swv')
    args = parser.parse_args()
    run_viz(**vars(args))
