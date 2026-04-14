import os
import sys
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_BASE: str
    OPENAI_API_KEY: str

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            print("CRITICAL ERROR: OPENAI_API_KEY is missing or empty in environment variables.")
            print("Please set OPENAI_API_KEY in your .env file or environment.")
            sys.exit(1)
        self.OPENAI_API_KEY = api_key

        self.OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        if not self.OPENAI_API_BASE.endswith("/"):
            self.OPENAI_API_BASE += "/"

        self.TIMEOUT_CONNECT = float(os.getenv("TIMEOUT_CONNECT", "5.0"))
        self.TIMEOUT_READ = float(os.getenv("TIMEOUT_READ", "300.0"))
        self.TIMEOUT_WRITE = float(os.getenv("TIMEOUT_WRITE", "20.0"))
        self.TIMEOUT_POOL = float(os.getenv("TIMEOUT_POOL", "10.0"))

        self.MAX_BODY_SIZE = int(os.getenv("MAX_BODY_SIZE", "15728640"))

        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


settings = Settings()
