# from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(find_dotenv(), override=True)


class Settings(BaseSettings):
    """Settings for the FastAPI application."""

    model_config = {
        "env_file": find_dotenv(),
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }
    CONFIG_DIR: str = "../../config"
    OPENAI_API_KEY: str
    GROQ_API_KEY: str
    DEEPSEEK_API_KEY: str
    OPENROUTER_API_KEY: str

    

SETTINGS = Settings()
