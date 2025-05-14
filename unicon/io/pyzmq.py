import os


def dump_json(states):
    import json
    data = {k: s.tolist() for k, s in states.items()}
    return json.dumps(data, indent=None, separators=(',', ':')).encode()


def load_json(msg):
    import json
    return json.loads(msg.decode())


def cb_send_pyzmq(keys=None, port=1337, host='*', robot_def=None, **states):
    keys = list(states.keys()) if keys is None else keys
    states = {k: states[k] for k in keys}
    import zmq
    dump_fn = dump_json
    context = zmq.Context()
    pub = context.socket(zmq.PUB)
    pub.setsockopt(zmq.CONFLATE, 1)
    # pub.setsockopt(zmq.SNDHWM, 2)
    # pub.setsockopt(zmq.SNDBUF, 2*1024)  # See: http://api.zeromq.org/4-2:zmq-setsockopt
    addr = os.environ.get('UNICON_PYZMQ_ADDR', f'tcp://{host}:{port}')
    print('pyzmq', addr, keys)
    pub.bind(addr)

    class cb:

        def __call__(self):
            msg = dump_fn(states)
            pub.send(msg)

        def __del__(self):
            pub.close()
            context.term()

    return cb()


def cb_recv_pyzmq(keys=None, port=1337, host='localhost', robot_def=None, **states):
    keys = list(states.keys()) if keys is None else keys
    states = {k: states[k] for k in keys}
    import zmq
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    # sub.setsockopt(zmq.RCVHWM, 2)
    sub.setsockopt(zmq.CONFLATE, 1)
    # sub.setsockopt(zmq.RCVBUF, 2*1024)
    addr = os.environ.get('UNICON_PYZMQ_ADDR', f'tcp://{host}:{port}')
    print('pyzmq', addr, keys)
    sub.connect(addr)
    sub.setsockopt(zmq.SUBSCRIBE, b'')
    load_fn = load_json
    n_again = 0

    class cb:

        def __call__(_):
            nonlocal n_again
            try:
                msg = sub.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                n_again += 1
                if n_again > 1000 and n_again % 100 == 0:
                    print('cb_recv_pyzmq timeout', n_again)
                return
            n_again = 0
            data = load_fn(msg)
            for k, v in data.items():
                s = states.get(k)
                if k is None:
                    continue
                s[:] = v

        def __del__(_):
            sub.close()
            context.term()

    return cb()
