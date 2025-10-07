from decouple import config
from dotenv import load_dotenv
from decimal import Decimal

def import_config(name, default=None, class_type=str, env_path=".env"):
    load_dotenv(env_path)
    if default is not None:
        value = config(name, default=default)
    else:
        value = config(name)
    if issubclass(class_type, bool):
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        else:
            raise ValueError(f"{value} is not {class_type} type")
    elif issubclass(class_type, int) and str(value).isnumeric():
        value = int(value)
    elif issubclass(class_type, list):
        value = [x for x in value.split(',') if x]
    elif issubclass(class_type, float) or issubclass(class_type, Decimal):
        value = class_type(value)
    return value

class Settings:
    TELEGRAM_BOT_TOKEN: str = import_config("TELEGRAM_BOT_TOKEN", default="", class_type=str)
    CHANNEL_ID: str = import_config("CHANNEL_ID", default="", class_type=str)
    DATABASE_URL: str = import_config("DATABASE_URL", default="sqlite:///./cejob.db", class_type=str)
    EMBED_MODEL: str = import_config("EMBED_MODEL", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", class_type=str)
    SIM_THRESHOLD: float = import_config("SIM_THRESHOLD", default="0.62", class_type=float)
    ADMIN_IDS: list = [int(x) for x in import_config("ADMIN_IDS", default="", class_type=list)]

settings = Settings()
