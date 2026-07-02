_signal_memo = {}


def signal_get_memo(x, idx=None, key=None, th=0.1, lerp=0.5, update=True):
    if idx is not None:
        x = float(x[idx])
    key = idx if key is None else key
    last_x = _signal_memo.get(key, 0)
    if update:
        lerp_r = 1 - lerp
        _signal_memo[key] = last_x * lerp + lerp_r * x
    return last_x - th


def signal_is_edge(x, idx=None, key=None, th=0.1, lerp=0.5):
    if idx is not None:
        x = float(x[idx])
    key = idx if key is None else key
    last_x = _signal_memo.get(key, 0)
    r = last_x <= th and x > th
    f = last_x >= th and x < th
    # print(_edge_memo, idx, key, r, last_x, x)
    lerp_r = 1 - lerp
    _signal_memo[key] = last_x * lerp + lerp_r * x
    return 1 if r else (-1 if f else 0)


def signal_is_rising(*args, **kwds):
    return signal_is_edge(*args, **kwds) > 0


def signal_is_falling(*args, **kwds):
    return signal_is_edge(*args, **kwds) < 0


def signal_is_clicks(x, idx=None, keys=None, clicks=1):
    if idx is not None:
        x = float(x[idx])
    key = idx if key is None else key
    last_x = _signal_memo.get(key, 0)
    r = last_x <= th and x > th
    f = last_x >= th and x < th
    # print(_edge_memo, idx, key, r, last_x, x)
    lerp_r = 1 - lerp
    _signal_memo[key] = last_x * lerp + lerp_r * x
    return 1 if r else (-1 if f else 0)
