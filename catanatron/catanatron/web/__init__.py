import os
from pathlib import Path

from flask import Flask, abort, send_from_directory
from flask_cors import CORS


def _serve_ui(app, ui_dist):
    """Serves a built UI from the API's own address. Routes like /games/<id> belong to
    the UI's router, so anything that isn't a file gets its page."""
    ui_dist = Path(ui_dist).resolve()

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def ui(path):
        if (ui_dist / path).is_file():
            return send_from_directory(ui_dist, path)
        # A missing script or API route should fail, not come back as a web page.
        if path.split("/")[0] in ("api", "assets"):
            abort(404)
        return send_from_directory(ui_dist, "index.html")


def create_app(test_config=None):
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)
    CORS(app)

    # ===== Load base configuration
    from catanatron.web.models import db, with_installed_driver

    database_url = os.environ.get("DATABASE_URL", "sqlite:///:memory:")
    database_url = with_installed_driver(database_url)
    secret_key = os.environ.get("SECRET_KEY", "dev")
    app.config.from_mapping(
        SECRET_KEY=secret_key,
        SQLALCHEMY_DATABASE_URI=database_url,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        # Hosted databases drop idle connections while they sleep.
        SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True},
        UI_DIST=os.environ.get("CATANATRON_UI_DIST"),
    )
    if test_config is not None:
        app.config.update(test_config)

    # ===== Initialize Database
    with app.app_context():
        db.init_app(app)
        db.create_all()

    # ===== Initialize Routes
    from . import api

    app.register_blueprint(api.bp)

    if app.config["UI_DIST"]:
        _serve_ui(app, app.config["UI_DIST"])

    return app
