def cb_dpg1_recv_send_close(
    title='unicon',
    width=960,
    height=480,
    plot_cfg=None,
    input_cfg=None,
    traj_len=100,
    **states,
):
    import threading
    import numpy as np
    # dpg 1.x
    import dearpygui.dearpygui as dpg
    dpg.create_context()
    plot_cfg = {} if plot_cfg is None else plot_cfg
    trajs = {}
    traj_x = np.arange(traj_len, dtype=np.float32)
    with dpg.window(width=width / 2, height=height):
        for k, v in plot_cfg.items():
            s = states[k]
            tag = f"plot_{k}"
            traj = np.zeros((traj_len, *s.shape), dtype=s.dtype)
            trajs[tag] = traj
            with dpg.plot(label=tag, width=320, height=240):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis)
                dpg.set_axis_limits(dpg.last_item(), 0, 100)
                y_tag = f'ax_{k}_y'
                dpg.add_plot_axis(dpg.mvYAxis, tag=y_tag)
                for idx in range(len(s)):
                    t = f'{tag}_{idx}'
                    dpg.add_line_series(traj_x, traj[:, idx].tolist(), label=t, parent=y_tag, tag=t)
                    print('add', t)

    input_cfg = {} if input_cfg is None else input_cfg
    input_states = {k: np.zeros_like(states[k]) for k in input_cfg}

    def slider_callback(sender, val, *args, _key, _idx):
        input_states[_key][_idx] = val

    with dpg.window():
        for k, v in input_cfg.items():
            s = states[k]
            tag = f"input_{k}"
            with dpg.group(tag=tag):
                for idx in range(len(s)):
                    label = f'{k}_{idx}'
                    dpg.add_slider_float(
                        label=label,
                        default_value=0.0,
                        min_value=-3.,
                        max_value=3.0,
                        callback=lambda s, val, *args, _k=k, _idx=idx: slider_callback(
                            s, val, *args, _key=_k, _idx=_idx),
                    )

    def gui_main_loop():
        dpg.create_viewport(title=title, width=width, height=height)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        if not dpg.is_viewport_ok():
            raise RuntimeError("Viewport was not created and shown.")
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
            time.sleep(0.01)

    th = threading.Thread(target=gui_main_loop, daemon=True)
    th.start()
    time.sleep(1)

    def cb_recv():
        for k, (tag, traj) in zip(plot_cfg, trajs.items()):
            s = states[k]
            traj[:-1] = traj[1:]
            traj[-1] = s
            for i in range(len(s)):
                t = f'{tag}_{i}'
                dpg.set_value(t, [traj_x, traj[:, i].tolist()])

    def cb_send():
        for k, v in input_states.items():
            states[k][:] = v

    def cb_close():
        dpg.destroy_context()

    return cb_recv, cb_send, cb_close


if __name__ == '__main__':
    import numpy as np
    import time
    states = {
        'states_recv': np.zeros(5),
        'states_send': np.zeros(3),
    }
    cb_recv, cb_send, cb_close = cb_dpg1_recv_send_close(
        **states,
        plot_cfg={
            'states_send': {},
            'states_recv': {},
        },
        input_cfg={
            'states_send': {},
        },
    )
    for _ in range(10000):
        s = states['states_recv']
        s[np.random.randint(len(s))] = np.random.randn()
        cb_recv()
        print('s', s)
        cb_send()
        time.sleep(0.02)
    cb_close()
