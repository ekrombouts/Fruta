from abc import ABC, abstractmethod
from typing import Any, List

from backend.config.settings import get_settings
from openai import AzureOpenAI, OpenAI
from sentence_transformers import SentenceTransformer


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def _initialize_client(self) -> Any:
        """Initialize the embedding client."""
        pass

    @abstractmethod
    def create_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    @abstractmethod
    def get_embedding_column_name(self) -> str:
        """Return the embedding column name."""
        pass


class OpenAIEmbedder(EmbeddingProvider):
    """Embedding provider implementation for OpenAI."""

    def __init__(self, settings):
        self.settings = settings
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        """Initialize the OpenAI client using the provided settings."""
        return OpenAI(api_key=self.settings.api_key)

    def create_embeddings(self, texts, **kwargs) -> List[List[float]]:
        """Generate embeddings using OpenAI's embedding API."""
        response = self.client.embeddings.create(
            input=texts, model=self.settings.embedding_model
        )
        return [r.embedding for r in response.data]

    def get_dimension(self) -> int:
        """Return the embedding dimension from settings."""
        return self.settings.embedding_dimension

    def get_embedding_column_name(self) -> str:
        """Return the column name for storing embeddings."""
        return self.settings.embedding_column


class AzureOpenAIEmbedder(EmbeddingProvider):
    """Embedding provider implementation for Azure OpenAI."""

    def __init__(self, settings):
        self.settings = settings
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        """Initialize the Azure OpenAI client using the provided settings."""
        return AzureOpenAI(
            api_key=self.settings.api_key,
            api_version=self.settings.api_version,
            azure_endpoint=self.settings.azure_endpoint,
        )

    def create_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings using Azure OpenAI's embedding API."""
        response = self.client.embeddings.create(
            input=texts, model=self.settings.embedding_model
        )
        return [r.embedding for r in response.data]

    def get_dimension(self) -> int:
        """Return the embedding dimension from settings."""
        return self.settings.embedding_dimension

    def get_embedding_column_name(self) -> str:
        """Return the column name for storing embeddings."""
        return self.settings.embedding_column


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Embedding provider implementation for Sentence Transformers."""

    def __init__(self, settings):
        self.settings = settings
        self.client = self._initialize_client()

    def _initialize_client(self) -> SentenceTransformer:
        """Initialize the SentenceTransformer model."""
        return SentenceTransformer(self.settings.embedding_model)

    def create_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings using the SentenceTransformer model."""
        return self.client.encode(texts, **kwargs).tolist()

    def get_dimension(self) -> int:
        """Return the embedding dimension from settings."""
        return self.settings.embedding_dimension

    def get_embedding_column_name(self) -> str:
        """Return the column name for storing embeddings."""
        return self.settings.embedding_column


class EmbeddingFactory:
    """Factory class to create embedding providers based on the specified provider."""

    def __init__(self, provider: str):
        """
        Initializes the EmbeddingFactory with the specified provider.

        Args:
            provider (str): The name of the embedding provider to use.

        Attributes:
            provider (str): The name of the embedding provider ("azure" or "sentence_transformer").
            settings (Any): The settings specific to the selected provider,
                retrieved from the global settings.
            embedding_provider (Any): The instance of the embedding provider
                created based on the specified provider.
        """
        self.provider = provider
        settings = get_settings()
        self.settings = getattr(settings.llm, provider)
        self.embedding_provider = self._create_provider()

    def _create_provider(self) -> EmbeddingProvider:
        """Create an embedding provider instance based on the provider name."""
        providers = {
            ## Comment or uncomment the providers you want to use
            "openai": OpenAIEmbedder,
            "azure": AzureOpenAIEmbedder,
            "sentence_transformer": SentenceTransformerEmbedder,
        }
        provider_class = providers.get(self.provider)
        if provider_class:
            return provider_class(self.settings)
        raise ValueError(f"Unsupported Embedding provider: {self.provider}")

    def create_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings using the selected embedding provider."""
        return self.embedding_provider.create_embeddings(texts, **kwargs)

    def get_dimension(self) -> int:
        """Return the embedding dimension from the selected provider."""
        return self.embedding_provider.get_dimension()

    def get_embedding_column_name(self) -> str:
        """Return the column name for storing embeddings."""
        return self.settings.embedding_column


# Example usage
if __name__ == "__main__":
    # Load settings
    settings = get_settings()
    # Get the embedding provider from settings
    provider = settings.embedding_provider
    # Create an embedding factory for the specified provider
    factory = EmbeddingFactory(provider=provider)
    # Example input texts
    texts = ["Hallo, ik heet Eva."]
    # Generate embeddings
    embeddings = factory.create_embeddings(texts)
    # Print the generated embeddings
    print(f"Embeddings voor {provider} \n {embeddings}\n")
    # Print the number of dimensions in the embeddings
    print(f"Aantal dimensies: {len(embeddings[0])}")
    # Print the embedding dimension from settings
    print(f"Aantal dimensies opgegeven in Settings: {factory.get_dimension()}")
    # Print the column name for embeddings
    print(f"Kolomnaam voor embeddings: {factory.get_embedding_column_name()}")
