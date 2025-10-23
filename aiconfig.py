import os
from typing import Protocol


class AIConfiguration(Protocol):
    api_base: str
    api_key: str
    api_engine: str
    api_type: str
    api_version: str
    stream: bool


class PIICheckerConfig(AIConfiguration):
    def __init__(self) -> None:
        self.api_base = os.environ.get("OPENAI_API_BASE", "https://azure.example.com")
        self.api_key = os.environ.get("OPENAI_API_KEY", "your_api_key")
        self.api_engine = os.environ.get("OPENAI_API_ENGINE", "engine")
        self.api_type = os.environ.get("OPENAI_API_TYPE", "azure")
        self.api_version = os.environ.get("OPENAI_API_VERSION", "2023-07-01")
        self.stream = False
