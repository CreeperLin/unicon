
def cb_recv_aiortc(
    keys=None,
    sig_port=9999,
    sig_host='localhost',
    ids=None,
    **states,
):
    import asyncio
    import threading
    from aiortc import RTCPeerConnection

    async def run_receiver():
        pc = RTCPeerConnection()

        @pc.on('track')
        def on_track(track):
            if track.kind == 'video':
                async def proc():
                    while True:
                        frame = await track.recv()
                        img = frame.to_ndarray(format='bgr24')
                asyncio.ensure_future(proc())

        await pc.wait_closed()

    receiver_thread = threading.Thread(target=start_receiver)
    receiver_thread.start()

    def start_receiver():
        asyncio.run(run_receiver())

    def cb():
        pass
    
    return cb


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<body>
<div id="videos"></div>

<script>
const CHANNELS = __CHANNELS__;

async function start() {
    const pc = new RTCPeerConnection();

    // Create one recvonly transceiver per channel
    for (let i = 0; i < CHANNELS; i++) {
        pc.addTransceiver("video", { direction: "recvonly" });
    }

    pc.ontrack = e => {
        const v = document.createElement("video");
        v.autoplay = true;
        v.playsInline = true;
        // v.srcObject = new MediaStream([e.track]);
        v.srcObject = e.streams[0];
        document.getElementById("videos").appendChild(v);
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const resp = await fetch("/offer", {
        method: "POST",
        body: JSON.stringify({
            sdp: pc.localDescription.sdp,
            type: pc.localDescription.type
        }),
        headers: { "Content-Type": "application/json" }
    });

    const answer = await resp.json();
    await pc.setRemoteDescription(answer);
}

start();
</script>
</body>
</html>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>WebRTC Multi‑Video</title>

    <style>
        button {
            padding: 8px 16px;
        }
        .option {
            margin-bottom: 8px;
        }
        #videos {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            max-width: 1280px;
        }
        video {
            width: 48%;
            background: #000;
        }
    </style>
</head>

<body>

<div class="option">
    <input id="use-stun" type="checkbox"/>
    <label for="use-stun">Use STUN server</label>
</div>

<button id="start" onclick="start()">Start</button>
<button id="stop" style="display: none" onclick="stop()">Stop</button>

<h2>Media</h2>
<div id="videos"></div>

<script>
let pc = null;

function negotiate() {
    // Request unlimited video tracks
    pc.addTransceiver('video', { direction: 'recvonly' });

    return pc.createOffer()
        .then((offer) => pc.setLocalDescription(offer))
        .then(() => {
            return new Promise((resolve) => {
                if (pc.iceGatheringState === 'complete') {
                    resolve();
                } else {
                    const checkState = () => {
                        if (pc.iceGatheringState === 'complete') {
                            pc.removeEventListener('icegatheringstatechange', checkState);
                            resolve();
                        }
                    };
                    pc.addEventListener('icegatheringstatechange', checkState);
                }
            });
        })
        .then(() => {
            return fetch('/offer', {
                body: JSON.stringify(pc.localDescription),
                headers: { 'Content-Type': 'application/json' },
                method: 'POST'
            });
        })
        .then((response) => response.json())
        .then((answer) => pc.setRemoteDescription(answer))
        .catch((e) => alert(e));
}

function start() {
    const config = { sdpSemantics: 'unified-plan' };

    if (document.getElementById('use-stun').checked) {
        config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
    }

    pc = new RTCPeerConnection(config);

    pc.addEventListener('track', (evt) => {
        const container = document.getElementById('videos');

        const video = document.createElement('video');
        video.autoplay = true;
        video.playsInline = true;
        video.srcObject = new MediaStream([evt.track]);

        container.appendChild(video);
    });

    document.getElementById('start').style.display = 'none';
    document.getElementById('stop').style.display = 'inline-block';

    negotiate();
}

function stop() {
    document.getElementById('stop').style.display = 'none';

    setTimeout(() => {
        if (pc) pc.close();
        pc = null;

        // Clear video elements
        const container = document.getElementById('videos');
        container.innerHTML = '';
    }, 300);
}
</script>

</body>
</html>
"""


def cb_send_aiortc(
    keys=None,
    web_host='0.0.0.0',
    web_port=8080,
    formats=None,
    **states,
):
    import asyncio
    import threading
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from aiortc.contrib.media import MediaRelay
    from av import VideoFrame
    from http.server import BaseHTTPRequestHandler, HTTPServer
    # from aiohttp import web

    loop = asyncio.new_event_loop()

    class Track(VideoStreamTrack):

        def __init__(self, states_img, format='bgr24', id=None):
            super().__init__()
            self.states_img = states_img
            self.format = format
            if id is not None:
                self._id = id

        async def recv(self):
            pts, time_base = await self.next_timestamp()
            video_frame = VideoFrame.from_ndarray(self.states_img, format=self.format)
            video_frame.pts = pts
            video_frame.time_base = time_base
            # print("sending frame", self.states_img.shape, video_frame, pts, time_base)
            return video_frame

    keys = list(states.keys()) if keys is None else keys
    formats = 'bgr24' if formats is None else formats
    formats = {k: formats for k in keys} if isinstance(formats, str) else formats
    relays = {k: MediaRelay() for k in keys}
    tracks = {k: Track(states[k], format=formats[k]) for k in keys}
    html_index = HTML_TEMPLATE.replace('__CHANNELS__', str(len(keys)))
    print('cb_send_aiortc', keys, relays, tracks)

    pcs = set()

    async def handle_offer(params):
        print('offer', params)
        ofr = RTCSessionDescription(
            sdp=params['sdp'], type=params['type']
        )
        pc = RTCPeerConnection()
        pcs.add(pc)

        @pc.on('connectionstatechange')
        async def _():
            if pc.connectionState == 'failed':
                await pc.close()
                pcs.discard(pc)

        # for k in keys:
        #     pc.addTransceiver('video', direction='sendonly')

        await pc.setRemoteDescription(ofr)

        for k in keys:
            subscribed = relays[k].subscribe(tracks[k])
            # print('subscribed', k, subscribed)
            pc.addTrack(subscribed)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        print("LOCAL ANSWER SDP:\n", pc.localDescription.sdp)
        print("TRANSCEIVERS:", pc.getTransceivers())

        return {
            "sdp": pc.localDescription.sdp, "type": pc.localDescription.type,
        }

    import json

    class Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self):
            # CORS preflight response
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Access-Control-Max-Age', '86400')
            self.end_headers()

        def do_GET(self):
            if self.path == '/':
                body = html_index.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != '/offer':
                self.send_error(404)
                return

            length = int(self.headers['Content-Length'])
            data = self.rfile.read(length)
            params = json.loads(data.decode('utf-8'))

            # WebRTC logic must run inside asyncio
            answer = asyncio.run(handle_offer(params))
            fut = asyncio.run_coroutine_threadsafe(handle_offer(params), loop)
            answer = fut.result() # dict

            # Send JSON response
            response = json.dumps(answer).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(response)


    # async def index(_):
    #     return web.Response(content_type='text/html', text=HTML)

    # async def offer(request):
    #     params = await request.json()

    # web.run_app(app, port=web_port)

    # async def start_web():
    #     app = web.Application()
    #     app.router.add_get('/', index)
    #     app.router.add_post('/offer', offer)
    #     runner = web.AppRunner(app)
    #     await runner.setup()
    #     site = web.TCPSite(runner, '0.0.0.0', web_port)
    #     await site.start()

    def rtc_thread():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def start_sender():
        # asyncio.run(start_web())
        server = HTTPServer((web_host, web_port), Handler)
        print(f'cb_send_aiortc web {web_host}:{web_port}')
        server.serve_forever()

    th_rtc = threading.Thread(target=rtc_thread, daemon=True)
    th_rtc.start()

    th_web = threading.Thread(target=start_sender, daemon=True)
    th_web.start()

    def cb():
        pass
    
    return cb
