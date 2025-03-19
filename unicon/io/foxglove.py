def cb_send_foxglove(**states,):
    import threading
    import asyncio
    import json
    import time
    from foxglove_websocket.server import FoxgloveServer

    server = None
    chan_id = None

    async def init():
        nonlocal server, chan_id
        server = FoxgloveServer(
            "0.0.0.0",
            8765,
            "example server",
            capabilities=["clientPublish", "services"],
            supported_encodings=["json"],
        )

        async def run_server():
            print('test')
            server.start()
            await server._task
            print('test2')

        _thread = threading.Thread(target=asyncio.run, args=(run_server(),))
        _thread.start()
        props = {}
        for k, v in states.items():
            props[k] = {
                'type': 'array',
                'items': {
                    'type': 'number',
                    'minItems': len(v),
                }
            }
        chan_id = await server.add_channel({
            "topic": 'unicon',
            "encoding": "json",
            "schemaName": "ExampleMsg",
            "schema": json.dumps({
                "type": "object",
                "properties": props,
            }),
            "schemaEncoding": "jsonschema",
        })

    asyncio.run(init())

    async def send():
        states_out = {k: v.tolist() for k, v in states.items()}
        # print(states_out)
        await server.send_message(
            chan_id,
            time.time_ns(),
            json.dumps(states_out).encode("utf8"),
        )

    def cb():
        asyncio.run(send())

    return cb
