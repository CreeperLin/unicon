def cb_send_graphite(
    inst=None,
    host='localhost',
    port=2003,
    addr=None,
    intv=1.0,
    keys=None,
    send_ts=False,
    **states,
):
    import socket
    import time
    import os
    import threading

    inst = os.environ.get('UNICON_INSTANCE_ID') if inst is None else inst
    inst = 'default' if inst is None else inst

    addr = os.environ.get('UNICON_GRAPHITE_ADDR') if addr is None else addr
    addr = f'{host}:{port}' if addr is None else addr
    host, port = addr.split(':')
    port = int(port)

    keys = list(states.keys()) if keys is None else keys
    states = {k: v for k, v in states.items() if k in keys}

    print('cb_send_graphite', addr, inst, keys)

    sock = socket.socket()
    sock.connect((host, port))

    last_data = {'payload': None}

    def worker():
        while True:
            payload = last_data['payload']
            if payload is not None:
                try:
                    sock.sendall(payload.encode('utf-8'))
                except Exception as e:
                    print('cb_send_graphite exc', e)
                last_data['payload'] = None
            time.sleep(intv)

    threading.Thread(target=worker, daemon=True).start()

    def cb():
        ts = int(time.time()) if send_ts else -1
        ts = str(ts)
        lines = []
        for k, v in states.items():
            name = f'{inst}.{k}.'
            for i, val in enumerate(v.tolist()):
                m = name + str(i)
                v = float(val)
                ln = f'{m} {v} {ts}'
                lines.append(ln)
        payload = '\n'.join(lines) + '\n'
        last_data['payload'] = payload

    return cb
