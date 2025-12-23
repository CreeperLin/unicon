def cb_chain(*cbs, next_on=[True], verbose=True):
    it = iter(cbs)
    cur_cb = None

    def cb():
        nonlocal cur_cb
        while True:
            if cur_cb is None:
                cur_cb = next(it, None)
                if verbose:
                    print('next in chain', cur_cb)
            if cur_cb is None:
                return True
            ret = cur_cb()
            if ret in next_on:
                cur_cb = None
                continue
            return ret

    cb._cbs = cbs
    return cb


cb_cat = cb_chain


def cb_zip(*cbs, return_on=[True], verbose=False):
    if not len(cbs):
        prev_cbs, last_cb = [], lambda: None
    else:
        prev_cbs, last_cb = cbs[:-1], cbs[-1]

    def cb():
        for cur_cb in prev_cbs:
            ret = cur_cb()
            if ret in return_on:
                if verbose:
                    print('return on', cur_cb, ret)
                return ret
        return last_cb()

    cb._cbs = cbs
    return cb


def cb_zip_all(*cbs, return_on=[True], verbose=False):
    cbs_set = list(cbs)

    def cb():
        for cur_cb in list(cbs_set):
            ret = cur_cb()
            if ret in return_on:
                if verbose:
                    print('return on', cur_cb, ret)
                    cbs_set.pop(cur_cb)
        if not len(cbs_set):
            return True

    cb._cbs = cbs
    return cb


def cb_loop(_cb=None, pred=None, max_steps=100, ret_val=True, recycle=False, verbose=True):
    i = -1

    def cb():
        nonlocal i
        i += 1
        if pred is not None and pred():
            if verbose:
                print('cb_loop pred', i)
            i = -1 if recycle else i
            return ret_val
        if max_steps > 0 and i >= max_steps:
            if verbose:
                print('cb_loop max_steps', i)
            i = -1 if recycle else i
            return ret_val
        if _cb is not None:
            return _cb()

    cb._cbs = [_cb]
    return cb


def cb_prod(outer, inner):
    run_outer = True

    def cb():
        nonlocal run_outer
        if run_outer:
            ret = outer()
            if ret is True:
                return ret
            run_outer = False
        ret = inner()
        if ret is True:
            run_outer = True

    cb._cbs = [outer, inner]
    return cb


def cb_print(keys=None, intvs=None, **states):
    from unicon.utils import pp_arr, get_ctx
    if not len(states):
        from unicon.states import states_get, states_get_specs
        states = {k: states_get(k) for k in states_get_specs()}
    if keys is not None:
        states = {k: v for k, v in states.items() if k in keys}
    if intvs is None:
        intvs = 1
    if isinstance(intvs, (int, float)):
        intvs = {k: intvs for k in states.keys()}
    dt = get_ctx().get('dt')
    if dt is not None:
        intvs = {k: (max(v // dt, 1) if isinstance(v, float) else v) for k, v in intvs.items()}
    pts = intvs.copy()
    steps = -1
    print('cb_print', list(states.keys()))

    def cb():
        nonlocal steps
        steps += 1
        flg = 1
        for k, v in states.items():
            pt = pts[k]
            if pt == 1:
                pt = intvs[k]
                if flg:
                    print(steps)
                    flg = 0
                print(k, pp_arr(v))
            else:
                pt = pt - 1
            pts[k] = pt

    return cb


def cb_noop():
    return lambda: None


def cb_break():
    return lambda: True


def cb_timeout(max_steps=1000, verbose=True):
    steps = 0

    def cb():
        nonlocal steps
        steps += 1
        if steps > max_steps:
            if verbose:
                print('timeout triggered', max_steps)
            return True

    return cb


def cb_rec(
    rec=None,
    verbose=True,
    rec_ts=True,
    rec_states=True,
    time_fn=None,
    reserve=100000,
    rec_keys=None,
    **states,
):
    import time
    import numpy as np
    time_fn = time.perf_counter if time_fn is None else time_fn
    rec_keys = states.keys() if rec_keys is None else rec_keys

    assert isinstance(rec, dict)
    print('reserving record', reserve)
    print('rec keys', rec_keys)
    ts_rec = np.zeros(reserve, dtype=np.float32) if rec_ts else None
    if rec_states:
        for k in rec_keys:
            states_rec = np.zeros((reserve, *states[k].shape), dtype=np.float32)
            rec[k] = states_rec
        print({k: v.shape for k, v in rec.items()})

    rec['ts'] = ts_rec
    rec['len'] = pt = -1

    def cb():
        nonlocal pt
        ts = time_fn()
        pt += 1
        if pt >= reserve:
            print('rec full')
            return True
        if verbose:
            if (pt + 1) % (reserve // 100) == 0:
                print('rec pt', pt)
        if rec_ts:
            ts_rec[pt] = ts
        if rec_states:
            for k in rec_keys:
                rec[k][pt] = states[k]
        rec['len'] = pt

    return cb


def cb_replay(
    dest=None,
    frames=None,
    verbose=False,
    keys=None,
    init_pt=0,
    num_frames=None,
    inds=None,
    repeats=1,
    loop=False,
    use_tqdm=False,
    **states,
):
    if dest is not None:
        states[('default' if inds is None else 'q')] = dest
    if not isinstance(frames, dict):
        frames = {list(states.keys())[0]: frames}
    keys = list(frames.keys()) if keys is None else keys
    keys = list(set(keys) & set(states.keys()))
    dof_keys = [k for k in keys if 'q' in k]
    non_dof_keys = [k for k in keys if 'q' not in k]
    rec_len = len(frames[keys[0]])
    num_frames = (rec_len - init_pt) if num_frames is None else num_frames
    end_pt = min(init_pt + num_frames, rec_len)
    print('cb_replay', num_frames, keys, dof_keys, non_dof_keys, inds)
    if use_tqdm:
        from tqdm import tqdm
        pbar = tqdm(total=end_pt)
        pbar.update(init_pt)

    pt = init_pt - 1
    rep = 0

    def cb():
        nonlocal pt, rep
        if rep == 0:
            pt += 1
            rep = repeats
            if use_tqdm:
                pbar.update()
        rep -= 1
        if verbose:
            if (pt + 1) % (num_frames // 10) == 0:
                print('rec pt', pt)
        if inds is not None:
            for k in dof_keys:
                # states[k][inds] = frames[k][pt][inds]
                states[k][inds] = frames[k][pt]
            for k in non_dof_keys:
                states[k][:] = frames[k][pt]
        else:
            for k in keys:
                states[k][:] = frames[k][pt]
        if pt >= end_pt - 1:
            print('end of play', end_pt)
            if not loop:
                if use_tqdm:
                    pbar.close()
                return True
            pt = init_pt - 1
            if use_tqdm:
                pbar.reset()
                pbar.update(init_pt)

    return cb


def cb_replay_state(
    dest,
    frames=None,
    num_frames=None,
    inds=None,
    repeats=1,
    loop=False,
):
    idx = 0
    iter_frames = iter(frames)
    inds = slice(None) if inds is None else inds
    rem = 0
    frame = None

    def cb():
        nonlocal idx, rem, frame, iter_frames
        if num_frames is not None and idx >= num_frames:
            return True
        if rem == 0:
            frame = next(iter_frames, None)
            idx += 1
            rem = repeats
        rem -= 1
        if frame is None:
            if loop:
                iter_frames = iter(frames)
                frame = next(iter_frames, None)
            if frame is None:
                return True
        dest[inds] = frame

    return cb


def cb_fixed_lat(
    _cb,
    fixed_lat=0.003,
    sleep_fn=None,
    time_fn=None,
):
    import time
    import unicon.utils
    sleep_fn = 'sleep_spin'
    sleep_fn = getattr(unicon.utils, sleep_fn) if isinstance(sleep_fn, str) else sleep_fn
    time_fn = time.perf_counter if time_fn is None else time_fn

    def cb():
        t0 = time_fn()
        ret = _cb()
        lat = time_fn() - t0
        if lat > fixed_lat:
            print('cb_fixed_lat timeout', lat, fixed_lat)
            return ret
        sleep_fn(t0, fixed_lat)
        return ret

    cb._cbs = [_cb]
    return cb


def cb_timer_set(
    ctx=None,
    time_fn=None,
):
    import time
    time_fn = time.perf_counter if time_fn is None else time_fn
    ctx = globals() if ctx is None else ctx

    def cb():
        ctx['t0'] = time_fn()

    return cb


def cb_timer_wait(
    ctx=None,
    wait=0.002,
    # sleep_fn=None,
    sleep_fn='sleep_block',
    time_fn=None,
    stats=False,
    verbose=False,
):
    import time
    import unicon.utils
    sleep_fn = getattr(unicon.utils, sleep_fn) if isinstance(sleep_fn, str) else sleep_fn
    time_fn = time.perf_counter if time_fn is None else time_fn
    ctx = globals() if ctx is None else ctx
    if stats:
        import numpy as np
        lats = np.zeros(2**16, dtype=np.float32)
    pt = 0
    timeouts = 0
    intv = 500

    def cb():
        nonlocal pt, timeouts
        t0 = ctx['t0']
        lat = time_fn() - t0
        pt += 1
        if stats:
            lats[pt - 1] = lat
            if pt % intv == 0:
                _lats = lats[pt - intv:pt]
                print('cb_timer_wait lat min/max/std/mean', pt, timeouts, np.min(_lats), np.max(_lats), np.std(_lats),
                      np.mean(_lats))
            if pt >= len(lats):
                pt = 0
        if wait <= 0:
            return
        if lat > wait:
            if verbose:
                print('cb_timer_wait timeout', pt, lat, wait)
            timeouts += 1
            return
        sleep_fn(t0, wait)

    return cb


def cb_wait_input(
    states_input,
    keys=None,
    inds=None,
    press_th=0.49,
    pred='any',
    clicks=2,
    press_intv=0.5,
    click_intv=1,
    prompt=False,
    verbose=True,
    input_keys=None,
):
    import numpy as np
    from unicon.utils import coalesce, get_ctx
    input_keys = coalesce(get_ctx().get('input_keys'), input_keys)
    inds = [input_keys.index(k) for k in keys] if inds is None else inds
    keys = [input_keys[i] for i in inds] if keys is None else keys
    _pt = 0
    _cur = 0
    _pressed = 0
    _last_pressed = None
    _last_clicked = None
    _last_prompt = 0
    pred = getattr(np, pred)
    import time

    def cb():
        nonlocal _cur, _pressed, _last_pressed, _last_clicked, _last_prompt, _pt
        if prompt:
            _pt += 1
            if time.monotonic() - _last_prompt > 2:
                print('waiting for input keys', round(_last_prompt), keys)
                _last_prompt = time.monotonic()
        if pred(states_input[inds] > press_th):
            if _pressed == 0:
                _pressed = 1
                _last_pressed = time.monotonic()
                if verbose:
                    print('cb_wait_input pressed')
        else:
            if _pressed == 0:
                return
            _pressed = 0
            if time.monotonic() - _last_pressed > press_intv:
                return
            lc = _last_clicked
            _last_clicked = time.monotonic()
            if lc is None or _last_clicked - lc > click_intv:
                _cur = 0
                # return
            if verbose:
                print('cb_wait_input clicked', _cur)
            _cur += 1
            if _cur >= clicks:
                print('cb_wait_input triggered', inds, _cur, clicks)
                return True

    return cb


def cb_copy_merge(
    merge_fn='sum',
    merge_keys=None,
    copy_nums=None,
    **states,
):
    import numpy as np
    copy_states = {}

    merge_fn = getattr(np, merge_fn) if isinstance(merge_fn, str) else merge_fn

    for k, n in zip(merge_keys, copy_nums):
        s = states.get(k)
        rs = np.zeros((n, *s.shape), dtype=s.dtype)
        copy_states[k] = rs

    def cb():
        for k in merge_keys:
            s = states.get(k)
            rs = copy_states.get(k)
            s[:] = merge_fn(rs, axis=0)

    cb.copy_states = copy_states
    return cb


def cb_if(_cb, pred):
    _pt = -1

    def cb():
        nonlocal _pt
        _pt += 1
        return _cb() if pred(_pt) else None

    cb._cbs = [_cb]
    return cb


def cb_scheduled(*cbs, sched='rr', sched_args=None, repeat=1):
    pt = 0
    cb_pt = -1
    cur_cb = None
    num_cbs = len(cbs)

    def sched_rr(n):
        from itertools import cycle
        yield from cycle(range(n))

    def sched_random(n, seed=None):
        import random
        random.seed(seed)
        while True:
            yield random.randint(0, n - 1)

    sched = locals().get(f'sched_{sched}') if isinstance(sched, str) else sched
    sched_args = {} if sched_args is None else sched_args
    gen = sched(num_cbs, **sched_args) if callable(sched) else gen

    def cb():
        nonlocal pt, cb_pt, cur_cb
        if pt == 0:
            pt = repeat
            cb_pt = next(gen)
            cur_cb = cbs[cb_pt]
        pt -= 1
        return cur_cb()

    return cb


def cb_loop_timed(
    _cb,
    num_steps=None,
    dt=0.02,
    dt_ofs=0,
    cb_dt=False,
    time_fn=None,
    sleep_fn='sleep_spin',
    stats=True,
):
    import time
    import numpy as np
    from unicon import utils
    from unicon.utils import latency
    sleep_fn = getattr(utils, sleep_fn) if isinstance(sleep_fn, str) else sleep_fn
    time_fn = time.perf_counter if time_fn is None else time_fn
    
    if int(dt_ofs) == 1:

        def test_fn():
            t0 = time_fn()
            sleep_fn(t0, dt)
            time_fn()

        lat = latency(test_fn, num_runs=int(3 // dt)) - dt
        print('sleep_fn lat', lat)
        dt_ofs = -lat
    print('dt_ofs', dt_ofs)

    def cb():
        frameno = -1
        timeouts = 0
        t_idles = 0
        if stats:
            t_ss = 0
            tis = []
        t_start = t0 = time_fn()
        while True:
            frameno += 1
            if num_steps is not None and frameno >= num_steps:
                break
            t_s = t0 + dt - time_fn()
            if t_s > 0:
                sleep_fn(t0, dt + dt_ofs)
            else:
                timeouts += 1
                if timeouts < 64:
                    print('### loop timeout:', frameno, t_s)
            if stats:
                t1 = time_fn()
            ret = _cb()
            if stats:
                t_ss += (t1 - t0)
                tis.append(t_s)
            if ret is True:
                break
            t0 += dt
            t_idles += t_s
            if cb_dt:
                t0 = time_fn()
        t_stop = time_fn()
        dura = t_stop - t_start
        print(f'{frameno} steps in {dura}s')
        if frameno > 0:
            avg = dura / frameno
            print(f'avg/timeout/idle: {avg}, {timeouts}, {t_idles / frameno}')
            if stats:
                tis = np.array(tis)[1:]
                print('t_idle min/max/avg/std', np.min(tis), np.max(tis), np.mean(tis), np.std(tis))
                t_ss_avg = t_ss / frameno
                print('t_ss avg/err', t_ss_avg, t_ss_avg - dt)
        return True

    cb._cbs = [_cb]
    return cb
