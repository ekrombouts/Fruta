# Beantwoordt vragen over één cliënt met RAG (retrieval-augmented generation):
# 1. de vraag wordt omgezet naar een embedding
# 2. de rapportages die het dichtst bij de vraag liggen worden opgehaald (pgvector)
# 3. de LLM beantwoordt de vraag met alleen het profiel en die rapportages als bron

from datetime import date
from typing import Any, Dict, List

import pandas as pd
from backend.config.settings import get_settings
from backend.database.fruta_queries import FrutaClient
from backend.llm.llm_factory import LLMService
from backend.services.context_retriever import ContextRetriever
from pydantic import BaseModel, Field

settings = get_settings()
llm = LLMService(provider=settings.llm_provider)


class ClientAntwoord(BaseModel):
    """Response model voor een vraag over een cliënt."""

    antwoord: str = Field(
        description="Het antwoord op de vraag, uitsluitend gebaseerd op het profiel en de rapportages"
    )
    bronnen: List[int] = Field(
        description="De nummers van de rapportages waarop het antwoord gebaseerd is. Leeg als het antwoord alleen uit het profiel komt"
    )
    voldoende_informatie: bool = Field(
        description="False als het profiel en de rapportages onvoldoende informatie bevatten om de vraag te beantwoorden"
    )


class ClientQA:
    def __init__(
        self,
        client: FrutaClient,
        start_date: date | str,
        end_date: date | str,
        limit: int = 8,
    ):
        self.client = client
        self.retriever = ContextRetriever(
            client_id=client.client_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
        # Tussenresultaten bewaren we, zodat de app kan laten zien wat er gebeurt
        self.context = pd.DataFrame()
        self.messages: List[Dict[str, str]] = []
        self.usage: Any = None

    def _retrieve(self, question: str) -> pd.DataFrame:
        # Stap 1 en 2: vraag embedden en de dichtstbijzijnde rapportages ophalen.
        # NB: de query gebruikt <-> (euclidische afstand). OpenAI-embeddings hebben lengte 1,
        # dus de volgorde is dan gelijk aan die van cosine similarity (<=>).
        query_vector = self.retriever.get_query_embedding(question)
        context = self.retriever.get_context(query_vector)
        # Chronologisch sorteren en nummeren, zodat de LLM naar bronnen kan verwijzen
        context = context.sort_values(by="date").reset_index(drop=True)
        context.insert(0, "nr", range(1, len(context) + 1))
        return context

    def _generate_prompt(self, question: str, context: pd.DataFrame) -> List[Dict[str, str]]:
        system_prompt = """Je bent een AI-assistent in een verpleeghuis. Je beantwoordt vragen van zorgmedewerkers over één cliënt.

# Richtlijnen:
1. Baseer je antwoord uitsluitend op het cliëntprofiel en de rapportages hieronder. Vul geen ontbrekende informatie aan en maak geen aannames.
2. De rapportages zijn opgehaald op basis van gelijkenis met de vraag; sommige kunnen irrelevant zijn. Gebruik alleen de relevante.
3. Noem bij je antwoord de nummers van de rapportages die je gebruikt hebt.
4. Staat het antwoord niet in het profiel of de rapportages? Zeg dat dan eerlijk.
5. Antwoord kort en feitelijk, in het Nederlands."""

        profile = f"""# Cliëntprofiel:
Naam: {self.client.naam}
Afdeling: {self.client.ward}
Diagnose: {self.client.diagnose}
Somatiek: {self.client.somatiek}
ADL: {self.client.adl}
Mobiliteit: {self.client.mobiliteit}
Gedrag: {self.client.gedrag}"""

        notes = "\n\n".join(
            f"[{row.nr}] {row.date}: {row.content}" for row in context.itertuples()
        )

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{profile}\n\n# Rapportages:\n{notes or '(geen rapportages gevonden)'}\n\n# Vraag:\n{question}",
            },
        ]

    def ask(self, question: str) -> ClientAntwoord:
        self.context = self._retrieve(question)
        self.messages = self._generate_prompt(question, self.context)
        # Stap 3: de LLM genereert een antwoord in de vorm van ClientAntwoord
        result, raw = llm.create_completion(ClientAntwoord, self.messages)
        self.usage = raw.usage
        return result


if __name__ == "__main__":
    import json

    # Test de ClientQA class
    client = FrutaClient("ca_01")
    qa = ClientQA(client, start_date="2024-01-01", end_date="2026-01-01", limit=8)
    result = qa.ask("Hoe gaat het met eten en drinken?")

    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    print(qa.context[["nr", "date", "distance"]])
    print(f"Tokens: {qa.usage.prompt_tokens} in, {qa.usage.completion_tokens} uit")
