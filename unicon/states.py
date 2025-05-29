import os
import numpy as np

_states_inds = {}
_states_arrs = {}
_states_specs = {}
_states_size = 0
_states_arr = None
_states_shm = None
_states_buf = None
_states_name = 'states'
_shm_reused = False
_tmp_dir = os.environ.get('TMP', '/tmp')

_dtype2str = {
    np.float32: 'f',
    np.float64: 'd',
    np.int32: 'i',
    # np.float: 'd',
}


def states_reset():
    global _states_size
    states_destroy(force=True)
    _states_inds.clear()
    _states_arrs.clear()
    _states_specs.clear()
    _states_size = 0


def states_new(name, numel, dtype=np.float32):
    assert _states_arr is None
    global _states_size
    assert name not in _states_inds
    size = numel
    st = _states_size
    ed = _states_size + size
    _states_inds[name] = slice(st, ed)
    _states_specs[name] = (st, ed, numel, _dtype2str[dtype])
    _states_size += size
    return st


def states_news(specs):
    for s in specs:
        states_new(*s)


def states_create(size, use_shm=False, max_size=0, reuse=False, clear=False, name=_states_name):
    num_floats = size
    num_bytes = num_floats * 4

    if not reuse and use_shm:
        states_destroy(force=True)

    if use_shm:
        global _states_shm, _shm_reused
        from multiprocessing import resource_tracker
        resource_tracker.register = lambda *args, **kwds: None
        resource_tracker.unregister = lambda *args, **kwds: None
        from multiprocessing import shared_memory
        print('shm', name, size, num_bytes)
        try:
            shm = shared_memory.SharedMemory(name=name)
            assert shm.buf.nbytes == num_bytes
            _shm_reused = True
            if clear:
                print('shm cleared')
                shm.buf[:] = bytes(len(shm.buf))
        except Exception as e:
            print('shm reuse failed', e)
            states_destroy(force=True)
            shm = shared_memory.SharedMemory(name=name, create=True, size=num_bytes)
        _states_shm = shm
        arr = np.ndarray((num_floats,), dtype=np.float32, buffer=shm.buf)
    else:
        buf = bytearray(num_bytes)
        # arr = np.zeros((num_floats,), dtype=np.float32)
        arr = np.ndarray((num_floats,), dtype=np.float32, buffer=buf)

    return arr


def states_destroy(force=False):
    global _states_arr, _states_shm, _states_buf
    if not force and _states_shm is None:
        return
    _states_arr = None
    _states_buf = None
    if _shm_reused:
        print('not destroying reused shm')
        return
    from multiprocessing import shared_memory
    try:
        shm = shared_memory.SharedMemory(name=_states_name) if _states_shm is None else _states_shm
        shm.close()
        shm.unlink()
        _states_shm = None
        states_remove_specs()
        print('shm destroy')
    except Exception as e:
        print('shm destroy failed', e)


def states_set_inds(inds):
    states_destroy()
    global _states_inds, _states_size
    _states_inds = inds
    _states_size = max([(inds.stop if isinstance(inds, slice) else max(inds)) for inds in _states_inds.values()])
    print('_states_size', _states_size)


def states_get_inds():
    return _states_inds


def states_save_specs(name=_states_name):
    import json
    specs = _states_specs
    path = os.path.join(_tmp_dir, name + '.json')
    with open(path, 'w') as f:
        json.dump(specs, f)


def states_load_specs(name=_states_name):
    import json
    path = os.path.join(_tmp_dir, name + '.json')
    with open(path, 'r') as f:
        specs = json.load(f)
    inds = {}
    for k, v in specs.items():
        inds[k] = slice(v[0], v[1])
    global _states_specs
    _states_specs = specs
    states_set_inds(inds)


def states_remove_specs(name=_states_name):
    path = os.path.join(_tmp_dir, name + '.json')
    if os.path.exists(path):
        os.remove(path)


def states_init(save=True, load=False, name=_states_name, **kwds):
    if load:
        states_load_specs(name=name)
    save = save and not load
    global _states_arr
    if _states_arr is None:
        _states_arr = states_create(_states_size, name=name, **kwds)
    if save:
        states_save_specs(name=name)


def states_get_ind(name):
    return _states_inds[name]


def states_get(name=None):
    assert _states_arr is not None
    if name is None:
        return _states_arr
        # return _states_buf
    arr = _states_arrs.get(name)
    if arr is None:
        inds = _states_inds.get(name)
        if inds is None:
            return None
        arr = _states_arr[inds]
        # arr = np.ndarray((), buffer=_states_buf[_states_inds[name]])
        _states_arrs[name] = arr
    return arr


def states_set(name, v):
    s = states_get(name)
    s[:] = v


def autowired(func):

    def wrapped(*args, **kwds):
        return func(*args, **kwds)

    return wrapped
