from pathlib import Path
import yaml


BASE_DIR = Path(__file__).resolve().parent.parent


def load_config():
    """
    Загружает config.yaml и добавляет вычисляемые пути проекта
    """
    config_path = BASE_DIR / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    # базовые директории
    paths = {
        "BASE_DIR": str(BASE_DIR),
        "DATA_DIR": str(BASE_DIR / "data"),
        "PROCESSED_DIR": str(BASE_DIR / "data" / "processed"),
        "CLEAN_DIR": str(BASE_DIR / "data" / "processed" / "clean"),
        "PARSED_DIR": str(BASE_DIR / "data" / "processed" / "parsed"),
        "CORPORA_DIR": str(BASE_DIR / "corpora"),
        "MODEL_DIR": str(BASE_DIR / "models"),
        "LOG_DIR": str(BASE_DIR / "logs"),
    }

    # дефолтные flask-настройки
    flask_defaults = {
        "SECRET_KEY": user_config.get("SECRET_KEY", "dev-secret-key"),
        "DEBUG": user_config.get("DEBUG", True),
    }

    # итоговый конфиг
    config = {}
    config.update(paths)
    config.update(flask_defaults)

    # пользовательские параметры (если есть в yaml)
    for key, value in user_config.items():
        if key not in config:
            config[key] = value

    return config
