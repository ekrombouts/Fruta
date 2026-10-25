import pandas as pd
from backend.config.settings import get_settings
from backend.database.db_connection import Database
from backend.llm.embedding_factory import EmbeddingFactory
from sqlalchemy import inspect, text
from tqdm import tqdm

settings = get_settings()


# Zorg ervoor dat de vector-extensie is ingeschakeld in de database
def ensure_vector_extension(engine):
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))


# Haal de kolomnamen van een tabel op
def get_existing_columns(engine, table_name: str):
    inspector = inspect(engine)
    return [col["name"] for col in inspector.get_columns(table_name)]


# Voeg een kolom toe aan de tabel
def add_column(engine, table_name: str, embedding_column: str, column_type: str):
    with engine.begin() as conn:
        conn.execute(
            text(
                f"ALTER TABLE {table_name} ADD COLUMN {embedding_column} {column_type};"
            )
        )


# Voeg de embedding-kolom toe aan de tabel als deze nog niet bestaat
def add_embedding_column_if_not_exists(
    engine, table_name: str, embedding_column: str, embedding_dimension: int
):
    ensure_vector_extension(engine)
    existing_columns = get_existing_columns(engine, table_name)
    if embedding_column not in existing_columns:
        add_column(
            engine, table_name, embedding_column, f"vector({embedding_dimension})"
        )


# Haal een batch rijen op waar de embedding-kolom NULL is en de 'text'-kolom niet leeg is
def fetch_rows_for_embedding(
    engine,
    table_name: str,
    text_column: str,
    embedding_column: str,
    id_column: str,
    batch_size: int,
):
    with engine.connect() as conn:
        result = pd.read_sql(
            text(
                f"SELECT {id_column}, {text_column} FROM {table_name} WHERE {embedding_column} IS NULL AND {text_column} IS NOT NULL AND {text_column} != '' LIMIT :limit"
            ),
            conn,
            params={"limit": batch_size},
        )
    return result


# Werk de embedding-kolom bij voor een batch rijen
def update_row_embeddings(
    engine, table_name: str, embedding_column: str, id_column: str, rows, embeddings
):
    with engine.begin() as conn:
        for i, (_, row) in enumerate(rows.iterrows()):
            conn.execute(
                text(
                    f"UPDATE {table_name} SET {embedding_column} = :embedding WHERE {id_column} = :id"
                ),
                {"embedding": embeddings[i], "id": row[id_column]},
            )


# Tel het totale aantal rijen waar de embedding-kolom NULL is
def count_null_embeddings(engine, table_name: str, embedding_column: str):
    with engine.connect() as conn:
        return (
            conn.execute(
                text(
                    f"SELECT COUNT(*) FROM {table_name} WHERE {embedding_column} IS NULL"
                )
            ).scalar()
            or 0
        )


# Werk de opgegeven tabel bij met embeddings voor rijen waar de embedding-kolom NULL is
def update_embeddings(
    engine,
    table_name: str,
    id_column: str,
    embed_factory: EmbeddingFactory,
    batch_size: int = 5,
):
    embedding_column = embed_factory.get_embedding_column_name()
    total = count_null_embeddings(engine, table_name, embedding_column)
    total_batches = (total + batch_size - 1) // batch_size
    pbar = tqdm(total=total_batches, desc="Embedding batches", unit="batch")

    while True:
        rows = fetch_rows_for_embedding(
            engine=engine,
            table_name=table_name,
            text_column="note",
            embedding_column=embedding_column,
            id_column=id_column,
            batch_size=batch_size,
        )
        if rows.empty:
            break

        embeddings = embed_factory.create_embeddings(texts=rows["note"].tolist())
        update_row_embeddings(
            engine=engine,
            table_name=table_name,
            embedding_column=embedding_column,
            id_column=id_column,
            rows=rows,
            embeddings=embeddings,
        )
        pbar.update(1)

    pbar.close()


# Initialiseer de databaseverbinding en embedding-factory
def initialize_database_and_factory(provider: str):
    db = Database()
    engine = db.get_engine()
    embed_factory = EmbeddingFactory(provider=provider)
    return engine, embed_factory


# Doorloop het hele proce voor de opgegeven tabel
def process_table(provider: str, table_name: str, id_column: str):
    engine, embed_factory = initialize_database_and_factory(provider)
    add_embedding_column_if_not_exists(
        engine=engine,
        table_name=table_name,
        embedding_column=embed_factory.get_embedding_column_name(),
        embedding_dimension=embed_factory.get_dimension(),
    )
    update_embeddings(
        engine=engine,
        table_name=table_name,
        id_column=id_column,
        embed_factory=embed_factory,
        batch_size=50,
    )


def main():
    tables = [
        ("records", "note_id"),
        # ("notes", "id"),
    ]
    for table_name, id_column in tables:
        process_table(
            provider=settings.embedding_provider,
            table_name=table_name,
            id_column=id_column,
        )


if __name__ == "__main__":
    main()
