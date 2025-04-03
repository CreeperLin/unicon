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
    publisher = context.socket(zmq.PUB)
    addr = os.environ.get('UNICON_PYZMQ_ADDR', f'tcp://{host}:{port}')
    print('pyzmq', addr, keys)
    publisher.bind(addr)

    class cb:

        def __call__(self):
            msg = dump_fn(states)
            publisher.send(msg)

        def __del__(self):
            publisher.close()
            context.term()

    return cb()


def cb_recv_pyzmq(keys=None, port=1337, host='localhost', robot_def=None, **states):
    keys = list(states.keys()) if keys is None else keys
    states = {k: states[k] for k in keys}
    import zmq
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    addr = os.environ.get('UNICON_PYZMQ_ADDR', f'tcp://{host}:{port}')
    print('pyzmq', addr, keys)
    socket.connect(addr)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    load_fn = load_json

    class cb:

        def __call__(self):
            try:
                msg = socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            data = load_fn(msg)
            for k, v in data.items():
                s = states.get(k)
                if k is None:
                    continue
                s[:] = v

        def __del__(self):
            socket.close()
            context.term()

    return cb()
