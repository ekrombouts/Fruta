# Centrale configuratie voor LLM- en embedding-instellingen (bijv. Azure, OpenAI, sentence transformers)

# Laadt omgevingsvariabelen en stelt standaardwaarden voor modellen/configuraties in voor gebruik in de app

# Niet-gebruikte providers moeten worden uitgecommentarieerd in LLMConfig om fouten tijdens het laden te voorkomen.

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


# Basisinstellingen voor alle LLM-providers
class LLMProviderSettings(BaseSettings):
    temperature: float = 0.0  # Bepaalt willekeurigheid in modeloutput
    max_tokens: Optional[int] = None  # Maximum aantal tokens in de respons
    max_retries: int = 3  # Aantal herhalingen voor mislukte verzoeken


# Instellingen specifiek voor OpenAI's LLMs. Erft van LLMProviderSettings.
class OpenAISettings(LLMProviderSettings):
    api_key: str | None = os.getenv("OPENAI_API_KEY")

    default_model: str = "gpt-6-luna"

    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    # De naam van de kolom die in de database wordt aangemaakt om de embeddings op te slaan
    embedding_column: str = "te3s_embedding"


# Instellingen specifiek voor Azure OpenAI's LLMs
class AzureOpenAISettings(LLMProviderSettings):
    api_key: str | None = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint: str | None = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version: str = "2024-02-01"

    default_model: str = "gpt-4o-mini"

    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    embedding_column: str = "te3s_embedding"


# Instellingen specifiek voor Anthropic's LLMs
class AnthropicSettings(LLMProviderSettings):
    api_key: str | None = os.getenv("ANTHROPIC_API_KEY")

    default_model: str = "claude-3-5-sonnet-20240620"
    max_tokens: int | None = 16384  # Claude vereist dat max_tokens is ingesteld


# Instellingen specifiek voor Llama LLMs
class LlamaSettings(LLMProviderSettings):
    api_key: str = "key"  # Placeholder API-sleutel (niet gebruikt)
    base_url: str = "http://localhost:11434/v1"  # Basis-URL voor Llama-server

    default_model: str = (
        "llama3.2:1b"  # Controleer beschikbare lokale modellen met `ollama list`
    )


# Instellingen specifiek voor Sentence Transformers
class StSettings(LLMProviderSettings):
    embedding_model: str = (
        "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"
    )
    embedding_dimension: int = 768
    embedding_column: str = "nfi_embedding"


# Hoofdconfiguratieklasse voor LLM-providers
class LLMConfig(BaseSettings):
    ## Voeg opmerkingen toe of verwijder ze om providers in of uit te schakelen

    openai: OpenAISettings = OpenAISettings()
    azure: AzureOpenAISettings = AzureOpenAISettings()
    anthropic: AnthropicSettings = AnthropicSettings()
    ollama: LlamaSettings = LlamaSettings()
    sentence_transformer: StSettings = StSettings()
