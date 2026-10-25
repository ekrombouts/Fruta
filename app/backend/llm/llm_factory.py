from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type

import instructor
from backend.config.settings import get_settings
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def _initialize_client(self) -> Any:
        """Initialize the client for the LLM provider."""
        pass

    @abstractmethod
    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        """Create a completion using the LLM provider."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""

    def __init__(self, settings):
        self.settings = settings
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        return instructor.from_openai(OpenAI(api_key=self.settings.api_key))

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Tuple[BaseModel, Any]:
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            # "none" houdt gedrag en kosten vergelijkbaar en laat temperature werken
            "reasoning_effort": kwargs.get("reasoning_effort", "none"),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_completion_tokens": kwargs.get(
                "max_completion_tokens", self.settings.max_tokens
            ),
            "response_model": response_model,
            "messages": messages,
        }
        return self.client.chat.completions.create_with_completion(**completion_params)


class AzureProvider(LLMProvider):
    """Azure OpenAI provider implementation."""

    def __init__(self, settings):
        self.settings = settings
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        return instructor.from_openai(
            AzureOpenAI(
                api_key=self.settings.api_key,
                api_version=self.settings.api_version,
                azure_endpoint=self.settings.azure_endpoint,
            )
        )

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Tuple[BaseModel, Any]:
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
        }
        return self.client.chat.completions.create_with_completion(**completion_params)


class OllamaProvider(LLMProvider):
    """Ollama provider implementation."""

    def __init__(self, settings):
        self.settings = settings
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        return instructor.from_openai(
            OpenAI(base_url=self.settings.base_url, api_key=self.settings.api_key),
            mode=instructor.Mode.JSON,
        )

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
        }
        return self.client.chat.completions.create_with_completion(**completion_params)


class LLMService:
    """
    A service for managing connections to different Large Language Model (LLM) providers.

    This class handles the initialization and configuration of the selected LLM provider automatically.

    Attributes:
        provider (str): The name of the LLM provider (e.g., "openai", "azure", "ollama").
        settings (Any): The configuration settings for the chosen provider.
        llm_provider (LLMProvider): The initialized provider instance.

    The provider name is required when creating an instance. Settings and the provider instance
    are determined automatically.
    """

    def __init__(self, provider: str):
        self.provider = provider
        settings = get_settings()
        self.settings = getattr(settings.llm, provider)
        self.llm_provider = self._create_provider()

    def _create_provider(self) -> LLMProvider:
        providers = {
            "openai": OpenAIProvider,
            "azure": AzureProvider,
            "ollama": OllamaProvider,
        }
        provider_class = providers.get(self.provider)
        if provider_class:
            return provider_class(self.settings)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Tuple[BaseModel, Any]:
        """
        Create a completion using the configured LLM provider.

        Args:
            response_model: Pydantic model class defining the expected response structure
            messages: List of message dictionaries containing the conversation
            **kwargs: Additional arguments to pass to the provider

        Returns:
            Tuple containing the parsed response model and raw completion

        Raises:
            TypeError: If response_model is not a Pydantic BaseModel
            ValueError: If the provider is not supported
        """
        if not issubclass(response_model, BaseModel):
            raise TypeError("response_model must be a subclass of pydantic.BaseModel")

        return self.llm_provider.create_completion(response_model, messages, **kwargs)
