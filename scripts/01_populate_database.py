from typing import Iterator

import pandas as pd
from backend.database.db_connection import Database
from datasets import Dataset, load_dataset
from sqlalchemy import inspect


# Laadt een Hugging Face-dataset en converteert deze naar een Pandas DataFrame
def dataset_to_df(dataset_name, split="all") -> pd.DataFrame | Iterator[pd.DataFrame]:
    ds = load_dataset(path=dataset_name, split=split)
    assert isinstance(ds, Dataset), "ds is not a Hugging Face Dataset"
    df = ds.to_pandas()
    # assert isinstance(df, pd.DataFrame), "df is not a Pandas DataFrame"
    return df


# Extraheert de tabelnaam uit de naam van de dataset
def get_table_name(dataset_name):
    return dataset_name.split("/")[-1].split("_")[-1]


# Schrijft een DataFrame naar de database in de opgegeven tabel
def write_to_db(df, table_name, engine, mode="replace"):
    df.to_sql(name=table_name, con=engine, if_exists=mode, index=False)
    print(f"Tabel '{table_name}' succesvol geschreven met mode '{mode}'.")


# Controleert of een tabel al bestaat en vraagt de gebruiker om een actie te kiezen
def handle_existing_table(inspector, table_name):
    if inspector.has_table(table_name):
        action = (
            input(f"Table '{table_name}' bestaat al. (r)eplace, (a)ppend of (c)ancel? ")
            .strip()
            .lower()
        )
        return (
            action if action in ["r", "a"] else "c"
        )  # Alleen geldige acties worden geaccepteerd
    return "r"  # Standaardactie als de tabel niet bestaat


# Verwerkt een enkele dataset: controleert of de tabel bestaat en schrijft de data naar de database
def process_dataset(dataset_name, engine):
    table_name = get_table_name(dataset_name)
    inspector = inspect(engine)
    action = handle_existing_table(inspector, table_name)
    if action == "c":  # Annuleer als de gebruiker dat aangeeft
        print(f"Overgeslagen: {table_name}")
        return

    # Laad de dataset en schrijf deze naar de database
    df = dataset_to_df(dataset_name)
    write_to_db(df, table_name, engine, mode="replace" if action == "r" else "append")


def main():
    db = Database()  # Maak databaseverbinding
    engine = db.get_engine()  # Haal de SQLAlchemy-engine op
    hf_datasets = [
        "ekrombouts/fruta_profiles",
        "ekrombouts/fruta_scenarios",
        "ekrombouts/fruta_records",
    ]

    # Controleer of alle tabellen al bestaan
    inspector = inspect(engine)
    all_exist = all(inspector.has_table(get_table_name(ds)) for ds in hf_datasets)
    if all_exist:
        print("De database is al gevuld.")  # Sla over als alles al bestaat
    else:
        # Verwerk elke dataset afzonderlijk
        for dataset_name in hf_datasets:
            process_dataset(dataset_name, engine)

    # Afterthought: Besloten om ook nog de Gardenia_notes dataset toe te voegen.
    # Deze dataset bevat voorbeelden van rapportages, niet gelinkt aan clientprofielen,
    # maar op veel voorkomende onderwerpen.

    df_notes = dataset_to_df(dataset_name="ekrombouts/gardenia_notes", split="all")
    # Controleer
    assert isinstance(df_notes, pd.DataFrame), "df_notes is not a Pandas DataFrame"

    # Add ID column
    df_notes["id"] = df_notes.index + 1
    table_name = "notes"
    action = handle_existing_table(inspector, table_name)
    if action == "c":  # Annuleer als de gebruiker dat aangeeft
        print(f"Overgeslagen: {table_name}")
    else:
        # Schrijf de notes dataset naar de database
        write_to_db(
            df_notes, table_name, engine, mode="replace" if action == "r" else "append"
        )


if __name__ == "__main__":
    main()
