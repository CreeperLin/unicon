import numpy as np


def cb_tf_noise(std=0.001, seed=11235, keys=None, **states):
    states = states if keys is None else {k: states[k] for k in keys}
    states_dims = [len(v) for v in states.values()]
    noise_dim = sum(states_dims)
    std = {k: std for k in states.keys()} if not isinstance(std, dict) else std
    noise_std = []
    for k, s in std.items():
        s = [s] * len(states[k]) if isinstance(s, (float, int)) else s
        s = np.array(s)
        noise_std.append(s)
    noise_std = np.concatenate(noise_std)
    rng = np.random.RandomState(seed)
    print('cb_tf_noise', std)

    def cb():
        noise = rng.randn(noise_dim) * noise_std
        pt = 0
        for k, v in states.items():
            pt1 = pt + len(v)
            v[:] = v + noise[pt:pt1]
            pt = pt1

    return cb


def cb_tf_affine(params=None, keys=None, **states):
    states = states if keys is None else {k: states[k] for k in keys}
    params = {k: params for k in states.keys()} if not isinstance(params, dict) else params
    ws = {}
    bs = {}
    for k, p in params.items():
        w, b = [np.array([x] * len(states[k]) if isinstance(x, (float, int)) else x) for x in p]
        ws[k] = w
        bs[k] = b
    print('cb_tf_affine', params)

    def cb():
        for k, v in states.items():
            w = ws[k]
            b = bs[k]
            v[:] = v * w + b

    return cb


def cb_tf_conv(params=2, keys=None, **states):
    states = states if keys is None else {k: states[k] for k in keys}
    params = {k: params for k in states.keys()} if not isinstance(params, dict) else params
    windows = {}
    ws = {}
    wsizes = {}
    for k, p in params.items():
        p = p if isinstance(p, (list, tuple)) else [p]
        wsize = p[0]
        w = p[1] if len(p) > 1 else (1 / wsize)
        w = np.array(w).reshape(-1, 1) if isinstance(w, list) else w
        ws[k] = w
        wsizes[k] = wsize
    print('cb_tf_conv', params)

    def cb():
        for k, v in states.items():
            window = windows.get(k)
            if window is None:
                window = np.zeros((wsizes[k], *v.shape))
                window[:] = v
                windows[k] = window
                continue
            window[:-1] = window[1:]
            window[-1] = v
            v_b = np.sum(window * ws[k], axis=0)
            v[:] = v_b

    return cb
