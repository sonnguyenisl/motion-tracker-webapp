"""SocketIO event handlers.

The heavy scoring happens in a background task kicked off by the
``POST /api/sessions`` route; these handlers manage the per-client room used
to stream progress back, and provide the ``stop`` hook from the design.
"""

from flask import request
from flask_socketio import join_room


def register_socket_events(socketio):
    @socketio.on("connect")
    def _on_connect():
        # Each client joins a room named after its socket id so the background
        # processing task can target progress events at exactly this browser.
        join_room(request.sid)
        socketio.emit("connected", {"room": request.sid}, to=request.sid)

    @socketio.on("join")
    def _on_join(_data=None):
        join_room(request.sid)
        socketio.emit("connected", {"room": request.sid}, to=request.sid)

    @socketio.on("stop")
    def _on_stop(_data=None):
        # Recording stopped on the client. The clip is uploaded over HTTP
        # (multipart) by camera.js; this is a lightweight acknowledgement so
        # the client can show "finalising…" immediately.
        socketio.emit(
            "progress",
            {"stage": "upload", "fraction": 0.02, "message": "Uploading recording…"},
            to=request.sid,
        )
