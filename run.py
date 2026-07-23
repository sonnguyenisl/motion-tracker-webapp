"""Aximove entry point.

Run with:  python app.py
"""

from app import create_app, socketio

app = create_app()

if __name__ == "__main__":
    # socketio.run wraps app.run and sets up the websocket server.
    socketio.run(app, host="0.0.0.0", port=1523,
                 allow_unsafe_werkzeug=True)
