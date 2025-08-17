from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

from backend.config.config import ALLOWED_ORIGINS
from backend.utils.logger import get_logger
from backend.routes.chat import bp as chat_bp

def create_app():
    load_dotenv()
    app = Flask(__name__)
    logger = get_logger("app")

    # CORS
    if not ALLOWED_ORIGINS or ALLOWED_ORIGINS == "*":
        CORS(app)
    else:
        origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
        CORS(app, resources={r"/*": {"origins": origins}})

    # Blueprints
    app.register_blueprint(chat_bp)

    @app.get("/health")
    def health():
        return {"ok": True, "ts": os.getenv("APP_TS", "n/a")}

    logger.info("Flask app initialized")
    return app

# WSGI 엔트리
app = create_app()

if __name__ == "__main__":
    # 개발용만 True, 운영은 False
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=debug)
