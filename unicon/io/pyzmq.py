import os
import numpy as np

BUF_SIZE = 128
RCVHWM = 32
SNDHWM = 32


def dump_json(states):
    import json
    data = {k: s.astype(np.float64).round(3).tolist() for k, s in states.items()}
    return json.dumps(data, indent=None, separators=(',', ':')).encode()


def load_json(msg):
    import json
    return json.loads(msg.decode())


def cb_send_pyzmq(keys=None, port=1337, norm_th=None, host='*', robot_def=None, **states):
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
    addr = os.environ.get('UNICON_PYZMQ_ADDR', f'tcp://{host}:{port}')
    print('cb_send_pyzmq', addr, keys, norm_th)
    pub.bind(addr)
    n_send = 0

    class cb:

        def __call__(self):
            nonlocal n_send
            _states = states
            if norm_th is not None:
                _states = {k: v for k, v in _states.items() if np.sum(np.abs(v)) / len(v) > norm_th}
                if not len(_states):
                    return
                print('_states', n_send, _states.keys())
            msg = dump_fn(_states)
            pub.send(msg)
            n_send += 1

        def __del__(self):
            pub.close()
            context.term()

    return cb()


def cb_recv_pyzmq(keys=None, port=1337, host='localhost', recv_modes='+', repeats=3, robot_def=None, **states):
    keys = list(states.keys()) if keys is None else keys
    states = {k: states[k] for k in keys}
    import zmq
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, RCVHWM)
    # sub.setsockopt(zmq.CONFLATE, 1)
    # sub.setsockopt(zmq.RCVBUF, BUF_SIZE)
    addr = os.environ.get('UNICON_PYZMQ_ADDR', f'tcp://{host}:{port}')
    print('cb_recv_pyzmq', addr, keys)
    sub.connect(addr)
    sub.setsockopt(zmq.SUBSCRIBE, b'')
    load_fn = load_json
    n_again = 0
    recv_modes = {} if recv_modes is None else recv_modes
    recv_modes = {k: recv_modes for k in keys} if isinstance(recv_modes, str) else recv_modes
    rem = repeats
    last_msg = None

    class cb:

        def __call__(_):
            nonlocal n_again, rem, last_msg
            try:
                msg = sub.recv(flags=zmq.NOBLOCK)
                last_msg = msg
                rem = repeats
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
            data = load_fn(msg)
            for k, v in data.items():
                s = states.get(k)
                if k is None:
                    continue
                mode = recv_modes.get(k)
                if mode is None:
                    s[:] = v
                elif mode == '+':
                    s[:] = s + v

        def __del__(_):
            sub.close()
            context.term()

    return cb()
