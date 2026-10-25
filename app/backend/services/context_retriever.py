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
    def __init__(self, client_id: str, start_date: str, end_date: str, limit: int = 5):
        self.client_id = client_id
        self.start_date = start_date
        self.end_date = end_date
        self.limit = limit

    def get_query_embedding(self, query: str) -> List[float]:
        return embedder.create_embeddings([query])[0]

    def get_context(self, query_vector: List[float]) -> pd.DataFrame:
        engine = Database().get_engine()
        vector_str = f"ARRAY{query_vector}::vector"

        sql = text(
            f"""
            SELECT r.client_id, achternaam, ward, date, note as content, {embedding_column} <-> {vector_str} AS distance
            FROM records r
            LEFT JOIN profiles c ON r.client_id = c.client_id
            WHERE r.client_id = :client_id
            AND date::date >= :start_date
            AND date::date <= :end_date
            ORDER BY distance
            LIMIT :limit
            """
        )

        with engine.connect() as conn:
            df = pd.read_sql(
                sql,
                conn,
                params={
                    "client_id": self.client_id,
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "limit": self.limit,
                },
            )
        return df


# example usage
if __name__ == "__main__":
    retriever = ContextRetriever(
        client_id="cc_01", start_date="2024-01-01", end_date="2026-01-01", limit=5
    )
    query_vector = retriever.get_query_embedding(
        "Geef aan of er een valincident is geweest"
    )
    context_df = retriever.get_context(query_vector)
    print(context_df)
