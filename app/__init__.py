"""Application factory + shared SocketIO instance."""

from flask import Flask
from flask_socketio import SocketIO

from config import Config

# Shared, app-unbound SocketIO instance (initialised in create_app).
socketio = SocketIO()


def create_app(config_class=Config):
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    app.config.from_object(config_class)
    config_class.ensure_dirs()

    # Database
    from . import db
    db.init_db(app)

    # HTTP routes
    from .routes import bp as main_bp
    app.register_blueprint(main_bp)

    # SocketIO
    socketio.init_app(app, async_mode=app.config["SOCKETIO_ASYNC_MODE"],
                       cors_allowed_origins="*")
    from .socket_events import register_socket_events
    register_socket_events(socketio)

    return app
