from typing import List

import pandas as pd
from backend.config.settings import get_settings
from backend.database.db_connection import Database
from backend.llm.embedding_factory import EmbeddingFactory
from sqlalchemy import text

settings = get_settings()
embedder = EmbeddingFactory(provider=settings.embedding_provider)
embedding_column = embedder.get_embedding_column_name()


class ContextRetriever:
    def __init__(self, limit: int = 5):
        self.limit = limit

    def get_query_embedding(self, query: str) -> List[float]:
        return embedder.create_embeddings([query])[0]

    def get_context(self, query_vector: List[float]) -> pd.DataFrame:
        engine = Database().get_engine()
        vector_str = f"ARRAY{query_vector}::vector"

        sql = text(
            f"""
            SELECT *, {embedding_column} <-> {vector_str} AS distance
            FROM notes
            ORDER BY distance ASC
            LIMIT :limit
            """
        )

        with engine.connect() as conn:
            df = pd.read_sql(
                sql,
                conn,
                params={
                    "limit": self.limit,
                },
            )
        return df


# example usage
if __name__ == "__main__":
    # Stap 1: Haal de valrisico-vector op uit de database
    engine = Database().get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT vector FROM topic_vectors WHERE onderwerp = 'valrisico'")
        )
        row = result.fetchone()
        if not row:
            raise ValueError("Geen vector gevonden voor onderwerp 'valrisico'")
        query_vector = row[0]  # Dit is een lijst van 768 floats (Postgres vector type)

    # Stap 2: Gebruik deze vector om context op te halen
    retriever = ContextRetriever(limit=5)
    context_df = retriever.get_context(query_vector)

    # Stap 3: Bouw prompt
    notes = "\n".join(["- " + n for n in context_df["note"]])

    print(prompt_instructie)
