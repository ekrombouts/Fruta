from functools import lru_cache

from backend.config.llm_config import LLMConfig
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):

    llm: LLMConfig = LLMConfig()
    # openai, azure of sentence_transformer
    embedding_provider: str = "openai"
    # openai, azure, anthropic, llama
    llm_provider: str = "openai"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Example retrieval of settings
if __name__ == "__main__":
    import json

    settings = get_settings()
    print(json.dumps(settings.llm.model_dump(mode="json"), indent=2))

    print(f"\nEmbedding provider: {settings.embedding_provider}")
