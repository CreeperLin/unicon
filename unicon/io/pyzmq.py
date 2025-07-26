import os
import numpy as np
import json

BUF_SIZE = 128
RCVHWM = 32
SNDHWM = 32


def dump_json(states):
    data = {k: s.astype(np.float64).round(3).tolist() for k, s in states.items()}
    return json.dumps(data, indent=None, separators=(',', ':')).encode()


def load_json(msg):
    return json.loads(msg.decode())


def cb_send_pyzmq(
    keys=None,
    port=1337,
    addr=None,
    norm_th=None,
    host='*',
    topic=None,
    send_key_map=None,
    verbose=False,
    **states,
):
    keys = list(states.keys()) if keys is None else keys
    states = {k: states[k] for k in keys}
    import zmq
    dump_fn = dump_json
    context = zmq.Context()
    pub = context.socket(zmq.PUB)
    # pub.setsockopt(zmq.CONFLATE, 1)
    pub.setsockopt(zmq.SNDHWM, SNDHWM)
    # See: http://api.zeromq.org/4-2:zmq-setsockopt
    # pub.setsockopt(zmq.SNDBUF, BUF_SIZE)
    # addr = os.environ.get('UNICON_PYZMQ_ADDR', f'tcp://{host}:{port}')
    addr = f'tcp://{host}:{port}' if addr is None else addr
    if topic is None or isinstance(topic, str):
        topic = {topic: keys}
    topic = {(k if k is None else k.encode()): v for k, v in topic.items()}
    print('cb_send_pyzmq', addr, keys, norm_th, topic)
    pub.bind(addr)
    n_send = 0
    send_key_map = {} if send_key_map is None else send_key_map

    class cb:

        def __call__(self):
            nonlocal n_send
            for tp, ks in topic.items():
                _states = {send_key_map.get(k, k): states[k] for k in ks}
                if norm_th is not None:
                    _states = {k: v for k, v in _states.items() if np.sum(np.abs(v)) / len(v) > norm_th}
                    if not len(_states):
                        return
                    print('_states', n_send, _states.keys())
                msg = dump_fn(_states)
                if tp is not None:
                    msg = tp + msg
                if verbose:
                    print('send', n_send, msg)
                pub.send(msg)
            n_send += 1

        def __del__(self):
            pub.close()
            context.term()

    return cb()


def cb_recv_pyzmq(
    keys=None,
    port=1337,
    host='localhost',
    addr=None,
    recv_modes='+',
    repeats=3,
    topic=None,
    recv_key_map=None,
    match_len=False,
    verbose=False,
    **states,
):
    keys = list(states.keys()) if keys is None else keys
    states = {k: states[k] for k in keys}
    import zmq
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, RCVHWM)
    # sub.setsockopt(zmq.CONFLATE, 1)
    # sub.setsockopt(zmq.RCVBUF, BUF_SIZE)
    addr = os.environ.get('UNICON_PYZMQ_ADDR') if addr is None else addr
    addr = f'tcp://{host}:{port}' if addr is None else addr
    print('cb_recv_pyzmq', addr, keys, topic)
    topic = '' if topic is None else topic
    if isinstance(topic, str):
        topic = {topic: keys}
    topic = {(k if k is None else k.encode()): v for k, v in topic.items()}
    for tp in topic:
        sub.setsockopt(zmq.SUBSCRIBE, tp)
    sub.connect(addr)
    load_fn = load_json
    n_again = -1
    recv_modes = {} if recv_modes is None else recv_modes
    recv_modes = {k: recv_modes for k in keys} if isinstance(recv_modes, str) else recv_modes
    rem = repeats
    last_msg = None
    ord_br = ord('{')
    recv_key_map = {} if recv_key_map is None else recv_key_map
    n_recv = 0

    class cb:

        def __call__(_):
            nonlocal n_again, rem, last_msg, n_recv
            try:
                msg = sub.recv(flags=zmq.NOBLOCK)
                last_msg = msg
                rem = repeats
                if n_again == -1:
                    print('cb_recv_pyzmq connected')
                if n_again > 1000:
                    print('cb_recv_pyzmq reconnected')
                n_again = 0
            except zmq.Again:
                n_again += 1
                if n_again > 1000 and n_again % 100 == 0:
                    print('cb_recv_pyzmq timeout', n_again)
                if rem <= 0 or last_msg is None:
                    return
                rem -= 1
                msg = last_msg
            idx = msg.index(ord_br)
            tp = msg[:idx]
            ks = topic.get(tp)
            if ks is None:
                return
            msg = msg[idx:]
            if not len(msg):
                return
            data = load_fn(msg)
            if verbose:
                print('recv', n_recv, data)
            for k in ks:
                v = data.get(k)
                if v is None:
                    continue
                k = recv_key_map.get(k, k)
                s = states.get(k)
                if s is None:
                    continue
                s_len = len(s)
                v_len = len(v)
                if match_len:
                    assert s_len == v_len
                mode = recv_modes.get(k)
                if mode is None:
                    s[:v_len] = v
                elif mode == '+':
                    s[:v_len] = s[:v_len] + v
            n_recv += 1

        def __del__(_):
            print('cb_recv_pyzmq destroy')
            sub.close()
            context.term()

    return cb()
