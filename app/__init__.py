from flask import Flask, render_template

from app.config import load_config
from app.db import init_sqlalchemy


def create_app():
    app = Flask(__name__)

    # загрузка конфига
    config = load_config()
    app.config.update(config)
    init_sqlalchemy(app)

    # регистрация blueprints (роутов)
    from app.routes.main import main_bp
    from app.routes.entries import entries_bp
    from app.routes.review import review_bp
    from app.routes.export import export_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(entries_bp, url_prefix="/entries")
    app.register_blueprint(review_bp, url_prefix="/review")
    app.register_blueprint(export_bp, url_prefix="/export")

    # обработчики ошибок
    register_error_handlers(app)

    return app


def register_error_handlers(app):

    @app.errorhandler(403)
    def forbidden(error):
        return render_template("errors/403.html"), 403

    @app.errorhandler(404)
    def not_found(error):
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def server_error(error):
        return render_template("errors/500.html"), 500
