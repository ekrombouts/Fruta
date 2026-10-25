# Mijn proleem is dit:
# Ik probeer semantisch relevante context te vinden in de rapportages met de zoekterm "vallen, valrisico, valincident",
# maar de resultaten vallen tegen: belangrijke teksten worden niet teruggevonden.
# In de UMAP-visualisatie zie ik dat valincidenten wél als cluster bij elkaar liggen.
# De structuur zit dus duidelijk in de data, maar mijn query weet die niet goed te pakken.

# Doel: een query vinden die relevante valincidentrapportages terugvindt met similarity search.

import ast
from typing import List, Optional

import numpy as np
import openai
import pandas as pd
from backend.config.settings import get_settings
from backend.database.db_connection import Database
from backend.llm.embedding_factory import EmbeddingFactory
from backend.llm.llm_factory import LLMService
from bertopic import BERTopic
from bertopic.representation import BaseRepresentation, KeyBERTInspired, OpenAI
from pydantic import BaseModel
from sqlalchemy import Engine, text
from tqdm import tqdm

settings = get_settings()

########################################################################
# Voorbereiden data
########################################################################
engine: Engine = Database().get_engine()
df: pd.DataFrame = pd.read_sql_table("notes", con=engine)
# df = df.iloc[0:1000]  # Beperk tot de eerste 1000 rijen voor snellere verwerking

# Haal de naam van de embedding kolom op basis van de provider.
factory: EmbeddingFactory = EmbeddingFactory(provider=settings.embedding_provider)
embedding_column: str = factory.get_embedding_column_name()

# De embeddings zijn opgeslagen als strings in de database,
# dus we moeten ze omzetten naar een lijst van floats
embedding_list: List[List[float]] = []
for embedding_str in tqdm(df[embedding_column], desc="Processing embeddings"):
    embedding_list.append(ast.literal_eval(embedding_str))
embs: np.ndarray = np.vstack(embedding_list)

# Extraheer de documenten
docs: List[str] = df["note"].tolist()
assert len(docs) == len(embs), "Lengte van docs en embeddings moet gelijk zijn."


####################################################################
# Llama representation model
#####################################################################


class LLaMARepresentation(BaseRepresentation):
    def __init__(self, llm_service, response_model):
        self.llm = llm_service
        self.response_model = response_model

    def transform(self, documents_per_topic, topics=None):
        labels = []
        for docs in documents_per_topic:
            sample_docs = docs[:5]
            prompt = (
                "Geef een kort label (max 5 woorden) dat de gemeenschappelijke inhoud van deze teksten samenvat:\n\n"
                + "\n\n".join(sample_docs)
            )
            messages = [
                {"role": "system", "content": "Je bent een Nederlandse taalexpert."},
                {"role": "user", "content": prompt},
            ]
            try:
                response_model_obj, _ = self.llm.create_completion(
                    response_model=self.response_model,
                    messages=messages,
                )
                label = response_model_obj.answer.strip()
            except Exception as e:
                label = "Onbekend onderwerp"
                print(f"Fout bij topic-labeling: {e}")
            labels.append(label)
        return labels


# class LLaMARepresentation:
#     def __init__(self, llm_service, response_model):
#         """
#         :param llm_service: Een instantie van jouw LLMService (uit de factory)
#         :param response_model: Een pydantic model (bijv. Answer) dat de response valideert
#         """
#         self.llm = llm_service
#         self.response_model = response_model

#     def transform(
#         self, documents_per_topic: List[List[str]], topics: Optional[List[int]] = None
#     ) -> List[str]:
#         labels = []

#         for docs in documents_per_topic:
#             sample_docs = docs[
#                 :5
#             ]  # max 5 documenten per topic, voorkomt te lange prompts

#             prompt = (
#                 "Geef een kort label (max 5 woorden) dat de gemeenschappelijke inhoud van deze teksten samenvat:\n\n"
#                 + "\n\n".join(sample_docs)
#             )

#             messages = [
#                 {"role": "system", "content": "Je bent een Nederlandse taalexpert."},
#                 {"role": "user", "content": prompt},
#             ]

#             try:
#                 response_model_obj, _ = self.llm.create_completion(
#                     response_model=self.response_model,
#                     messages=messages,
#                 )
#                 label = response_model_obj.answer.strip()
#             except Exception as e:
#                 label = "Onbekend onderwerp"  # fallback
#                 print(f"Fout bij topic-labeling: {e}")

#             labels.append(label)

#         return labels


class Answer(BaseModel):
    answer: str


########################################################################
# Train BERTopic
########################################################################

# Het representatiemodel zorgt voor het labelen van de topics.
representation_model = KeyBERTInspired()

# client = openai.OpenAI()
# representation_model = OpenAI(client, model="gpt-4o-mini", chat=True)


# llm = LLMService(provider=settings.llm_provider)
# representation_model = LLaMARepresentation(llm, response_model=Answer)

topic_model = BERTopic(
    language="dutch",
    verbose=True,
    calculate_probabilities=True,
    representation_model=representation_model,
    nr_topics=50,
    # Kies het juiste embedding model op basis van de gekozen provider in settings
    embedding_model=getattr(settings.llm, settings.embedding_provider).embedding_model,
)

topics, probs = topic_model.fit_transform(
    documents=docs,
    embeddings=embs,  # Gebruikt de embeddings die we eerder hebben gemaakt
)

assert (
    probs is not None
), "probs is None. Heb je calculate_probabilities=True ingesteld?"

########################################################################
# Resultaten
########################################################################

df_topic_info = topic_model.get_topic_info()
df_doc_info = topic_model.get_document_info(docs)

df_representative_docs = topic_model.get_representative_docs()

# # Visualisatie
topic_model.visualize_topics()
topic_model.visualize_hierarchy()

# fall_docs = topic_model.get_representative_docs(28)
# df_fall = df[df["note"].isin(fall_docs)]
# fall_vectors = np.vstack(df_fall[embedding_column].apply(ast.literal_eval).tolist())

# print(fall_vectors[:, :5])
# fall_vector = fall_vectors.mean(axis=0)
# print(fall_vector[:5])

# df_topic_vectors = pd.DataFrame(
#     [
#         {
#             "onderwerp": "valrisico",
#             "vector": fall_vector.tolist().__str__(),
#         }
#     ]
# )


# with engine.begin() as conn:
#     conn.execute(text("DROP TABLE IF EXISTS topic_vectors"))
#     conn.execute(
#         text(
#             "CREATE TABLE topic_vectors (onderwerp TEXT PRIMARY KEY, vector VECTOR(768))"
#         )
#     )

# df_topic_vectors.to_sql("topic_vectors", con=engine, if_exists="append", index=False)
