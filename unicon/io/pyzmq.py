def dump_json(states):
    import json
    data = {k: s.tolist() for k, s in states.items()}
    return json.dumps(data, indent=None, separators=(':', ',')).encode()


def load_json(msg):
    import json
    return json.loads(msg.decode())


def cb_send_pyzmq(**states):
    import zmq
    dump_fn = dump_json
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind('tcp://*:1337')

    class cb:

        def __call__(self):
            msg = dump_fn(states)
            publisher.send(msg)

        def __del__(self):
            publisher.close()
            context.term()

    return cb()


def cb_recv_pyzmq(**states):
    import zmq
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5556")
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    load_fn = load_json

    class cb:

        def __call__(self):
            msg = socket.recv()
            data = load_fn(msg)
            for k, v in data.items():
                states[k][:] = v

        def __del__(self):
            socket.close()
            context.term()

    return cb()
