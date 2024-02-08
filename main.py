import os
from importlib import import_module

from flask import Flask

from settings import Settings


def create_app(env):
    app = Flask(__name__, static_url_path="")

    for version in os.listdir("./"):
        if os.path.isdir(version) and version[0] == "v":
            api_version = import_module("{}.app".format(version))
            app.register_blueprint(
                getattr(api_version, "api_bp"), url_prefix="/" + version
            )

    return app


cfg = Settings()

cfg.set_env(os.environ["MODE_DEPLOY"])
cfg.load_model()

app = create_app("Production")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 9191)), debug=True)
