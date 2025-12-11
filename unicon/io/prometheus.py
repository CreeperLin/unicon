def cb_send_prometheus(
    inst=None,
    host='localhost',
    port=9091,
    addr=None,
    test=True,
    intv=1.0,
    keys=None,
    job='default',
    **states,
):
    import requests
    import time
    import os
    import threading

    inst = os.environ.get('UNICON_INSTANCE_ID') if inst is None else inst
    inst = 'default' if inst is None else inst

    addr = os.environ.get('UNICON_PROMETHEUS_ADDR') if addr is None else addr
    addr = f'http://{host}:{port}' if addr is None else addr
    # url = f'{addr}/metrics/job/{inst}'
    url = f'{addr}/metrics/job/{job}/instance/{inst}'

    keys = list(states.keys()) if keys is None else keys
    states = {k: v for k, v in states.items() if k in keys}

    print('cb_send_prometheus', url, keys)

    last_data = {"payload": None}

    def post(data):
        try:
            res = requests.post(url, data=data, headers={'Content-Type': 'text/plain'})
            if res.status_code != 200:
                print('cb_send_prometheus code', res.status_code, res.text)
        except Exception as e:
            print('cb_send_prometheus exc', e)

    def worker():
        while True:
            if last_data["payload"] is not None:
                post(last_data["payload"])
                last_data["payload"] = None
            time.sleep(intv)

    if test:
        post('\n')

    threading.Thread(target=worker, daemon=True).start()

    def cb():
        lines = []
        for k, v in states.items():
            metric_name = f'{k}'
            for i, val in enumerate(v.tolist()):
                val = float(val)
                lines.append(f'{metric_name}{{index="{i}"}} {val}')
        data = '\n'.join(lines) + '\n'
        last_data["payload"] = data

    return cb
