from __future__ import annotations
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    app_name: str = "qtinfra-demo"
    db_url: str = Field(default="sqlite+aiosqlite:///:memory:")
    log_level: str = "INFO"
    enable_metrics: bool = True

    model_config = {
        "env_prefix": "QTINFRA_",
        "extra": "ignore",
    }

settings = Settings()