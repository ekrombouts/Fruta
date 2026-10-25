import ast
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from backend.config.settings import get_settings
from backend.database.db_connection import Database
from backend.llm.embedding_factory import EmbeddingFactory
from backend.llm.llm_factory import LLMService
from pydantic import BaseModel, Field
from sqlalchemy import text

settings = get_settings()
embedder = EmbeddingFactory(provider=settings.embedding_provider)
embedding_column = embedder.get_embedding_column_name()
llm = LLMService(provider=settings.llm_provider)


class LLMResponse(BaseModel):
    """Response model voor valrisico-inschattingen."""

    gedachtengang: List[str] = Field(
        description="Overwegingen/gedachtengang van de AI-assistent bij het genereren van het antwoord"
    )
    valincident: bool = Field(
        description="een gedocumenteerd valincident in het dossier"
    )

    datum_laatste_valincident: date = Field(
        description="Datum van het laatste valincident in het dossier. 1999-01-01 als er geen valincidenten zijn gedocumenteerd"
    )

    class ValrisicoEnum(str, Enum):
        ZEER_HOOG = "Zeer hoog"
        HOOG = "Hoog"
        GEMIDDELD = "Gemiddeld"
        LAAG = "Laag"
        ONBEKEND = "Onbekend"

    valrisico: ValrisicoEnum = Field(
        description="Een inschatting van het valrisico van de cliënt"
    )


class FallRiskAnalyzer:
    def __init__(self, client_id: str, start_date: str, end_date: str, limit: int = 5):
        self.client_id = client_id
        self.start_date = start_date
        self.end_date = end_date
        self.limit = limit
        self.context = pd.DataFrame()

    def _get_query_embedding(self, query: str) -> List[float]:
        # Generate an embedding for the query using the configured embedder
        return embedder.create_embeddings([query])[0]

    def _get_topic_embedding(self, topic: str) -> List[float]:
        # Retrieve the embedding for the specified topic from the database
        engine = Database().get_engine()
        with engine.begin() as conn:
            result = conn.execute(
                text("SELECT vector FROM topic_vectors WHERE onderwerp = :onderwerp"),
                {"onderwerp": topic},
            ).scalar()
        return ast.literal_eval(result)

    def _get_context(self, query_vector: List[float]) -> pd.DataFrame:
        # Retrieve relevant records from the database based on the query vector
        engine = Database().get_engine()
        vector_str = f"ARRAY{query_vector}::vector"

        sql = text(
            f"""SELECT r.client_id, achternaam, ward, date, note as content, {embedding_column} <-> {vector_str} AS distance
            FROM records r
            LEFT JOIN profiles c ON r.client_id = c.client_id
            WHERE r.client_id = :client_id
            AND date::date >= :start_date
            AND date::date <= :end_date
            ORDER BY distance
            LIMIT :limit"""
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

    def _dataframe_to_json(self, df: pd.DataFrame, columns: List[str]) -> str:
        return df[columns].to_json(orient="records", indent=2)

    def _generate_prompt(self, context: pd.DataFrame) -> List[Dict[str, str]]:
        system_prompt = """Je bent een AI-assistent voor een verpleeghuissysteem dat de zorg ondersteunt in het genereren van zorgplannen. 
Je taak is om een inschatting te maken van het valrisico op basis van relevante rapportages uit het cliëntdossier.

# Richtlijnen:
1. Maak een inschatting van het valrisico op basis van de rapportages. NB Angst en verwardheid op zich zijn niet altijd een indicatie van een verhoogd valrisico.
2. Geef aan of de cliënt een valincident heeft gehad en wanneer dit was. NB een bijna valincident is geen valincident.
3. De context wordt op basis van cosine similarity opgehaald; sommige informatie kan ontbreken of irrelevant zijn. Gebruik uitsluitend de relevante rapportages voor het genereren van je antwoord.
4. Vul geen ontbrekende informatie aan en maak geen aannames."""

        context_str = (
            "[]"
            if context.empty
            else self._dataframe_to_json(
                context, ["content", "date", "achternaam", "ward"]
            )
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"# Rapportages:{context_str}"},
        ]

    def _call_llm(self, messages: List[Dict[str, str]]) -> Tuple[LLMResponse, Any]:
        response_parsed, response_raw = llm.create_completion(LLMResponse, messages)
        return LLMResponse(**response_parsed.model_dump()), response_raw

    def analyze(self) -> LLMResponse:
        query_vector = self._get_query_embedding(
            "valrisico, valincidenten. Client is gevallen."
        )

        self.context = self._get_context(query_vector)
        messages = self._generate_prompt(self.context)
        result, _ = self._call_llm(messages)
        return result


if __name__ == "__main__":
    import json

    # Test the FallRiskAnalyzer class
    client_id = "cc_01"
    start_date = "2024-01-01"
    end_date = "2026-01-01"
    limit = 5

    fra = FallRiskAnalyzer(
        client_id=client_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    result = fra.analyze()
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False, default=str))
