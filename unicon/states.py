import os
import numpy as np
import inspect

_states_arrs = {}
_states_specs = {}
_states_size = 0
_states_shm = None
_states_buf = None
_states_name = 'states'
_shm_reused = False
_tmp_dir = os.environ.get('TMP', '/tmp')

_dtype2size = {
    np.float32: 4,
    np.float64: 8,
    np.int32: 4,
}
_dtype2str = {
    np.float32: 'f',
    np.float64: 'd',
    np.int32: 'i',
}
_str2dtype = {v: k for k, v in _dtype2str.items()}


def states_reset():
    global _states_size
    states_destroy(force=True)
    _states_arrs.clear()
    _states_specs.clear()
    _states_size = 0


def states_new(name, shape, dtype=np.float32):
    assert _states_buf is None
    global _states_size
    assert name not in _states_specs
    shape = (shape,) if isinstance(shape, int) else shape
    numel = 1
    for x in shape:
        numel *= x
    size = numel * _dtype2size[dtype]
    st = _states_size
    ed = _states_size + size
    _states_specs[name] = (st, ed, numel, shape, _dtype2str[dtype])
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
        buf = shm.buf
    else:
        buf = bytearray(num_bytes)

    return buf


def states_destroy(force=False):
    global _states_shm, _states_buf
    if not force and _states_shm is None:
        return
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


def states_get_specs():
    return _states_specs


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
    global _states_specs, _states_size
    _states_specs = specs
    _states_size = max([spec[1] for spec in specs.values()])
    print('_states_size', _states_size)
    states_destroy()


def states_remove_specs(name=_states_name):
    path = os.path.join(_tmp_dir, name + '.json')
    if os.path.exists(path):
        os.remove(path)


def states_init(save=True, load=False, states_name=_states_name, **kwds):
    if load:
        states_load_specs(name=states_name)
    print('states_init', _states_specs)
    save = save and not load
    global _states_buf
    if _states_buf is None:
        _states_buf = states_create(_states_size, name=states_name, **kwds)
    if save:
        states_save_specs(name=states_name)


def states_get(name=None):
    assert _states_buf is not None
    if name is None:
        return _states_buf
    arr = _states_arrs.get(name)
    if arr is None:
        spec = _states_specs.get(name)
        if spec is None:
            return None
        st, ed, numel, shape, dtype = spec
        dtype = _str2dtype[dtype]
        arr = np.frombuffer(_states_buf, dtype=dtype, count=numel, offset=st).reshape(*shape)
        _states_arrs[name] = arr
    return arr


def states_set(name, v):
    s = states_get(name)
    s[:] = v


def autowired(func, states=None):
    sig = inspect.signature(func)
    skwds = {}
    states_prefix = 'states_'
    for k, v in sig.parameters.items():
        if not k.startswith(states_prefix):
            continue
        s = None
        if states is not None:
            s = states.get(k)
        if s is None:
            s = states_get(k.replace(states_prefix, ''))
        if s is None:
            if v.default is inspect._empty:
                raise ValueError(f'missing states param {k} for func {func}')
        else:
            skwds[k] = s
    states_var = sig.parameters.get('states')
    if states_var is not None and states_var.kind == states_var.VAR_KEYWORD:
        if states is not None:
            skwds.update(states)
        else:
            skwds.update({states_prefix + k: states_get(k) for k in _states_specs})

    def wrapped(*args, **kwds):

        return func(*args, **kwds, **skwds)

    return wrapped
