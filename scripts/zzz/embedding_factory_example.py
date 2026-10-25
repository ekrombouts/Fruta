from backend.config.settings import get_settings
from backend.llm.embedding_factory import EmbeddingFactory

settings = get_settings()

ef = EmbeddingFactory(provider=settings.llm_provider)
print(ef.create_embeddings(["Ik heet Eva"]))

ef = EmbeddingFactory(provider="sentence_transformer")
print(ef.create_embeddings(["Ik heet Eva"]))
