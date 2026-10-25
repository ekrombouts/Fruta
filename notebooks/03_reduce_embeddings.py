import ast
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from backend.config.settings import get_settings
from backend.database.db_connection import Database
from backend.llm.embedding_factory import EmbeddingFactory
from backend.llm.embedding_visualizer import create_interactive_plot
from umap import UMAP

settings = get_settings()

# Laad de dataset uit de database
print("Loading dataset...")
engine = Database().get_engine()
df = pd.read_sql_table("notes", con=engine)

print("Processing embeddings...")
factory = EmbeddingFactory(provider=settings.embedding_provider)
embedding_column = factory.get_embedding_column_name()

# De embeddings zijn opgeslagen als strings in de database, dus we moeten ze omzetten naar een lijst van floats
embedding_list: List[List[float]] = (
    df[embedding_column].apply(ast.literal_eval).tolist()
)
# En omzetten naar een numpy array
embs: np.ndarray = np.vstack(embedding_list)

# Reduceer de embeddings naar 2D met UMAP
print("Reducing embeddings with UMAP...")
UMAP_PARAMS = {
    "n_components": 2,  # Reduceert naar 2D voor visualisatie. 3D is ook mogelijk, maar moeilijker te interpreteren
    "n_neighbors": 12,  # Default: 15. Lagere waarden leiden tot meer lokale structuren, hogere waarden tot meer globale
    "min_dist": 0.05,  # Default: 0.1. Lagere waarde = dichtere clusters en meer gescheiden groepen
    "random_state": 6,  # Zorgt voor reproduceerbaarheid
}

# Train UMAP.
reducer = UMAP(**UMAP_PARAMS)
reduced_embs = reducer.fit_transform(embs)

# Voeg toe aan df. Gebruikelijk is om de gereduceerde embeddings in een list te zetten
df["reduced_embeddings"] = reduced_embs.tolist()

# Visualiseer
print("Creating interactive plot...")
fig = create_interactive_plot(
    data=df,
    embedding_column="reduced_embeddings",
    color_column="category",
    hover_column="note",
    title="2D Visualization of Reduced Embeddings using UMAP",
    caption=f"Parameters: n_neighbors={UMAP_PARAMS['n_neighbors']}, min_dist={UMAP_PARAMS['min_dist']}, random_state={UMAP_PARAMS['random_state']}",
)

# Sla plot en model op
print("Saving plot and model...")
root = Path(__file__).resolve().parents[1]
model_folder = root / "app" / "backend" / "models"
model_folder.mkdir(parents=True, exist_ok=True)
joblib.dump(reducer, model_folder / "notes_umap_2d.pkl")

plot_folder = root / "output"
plot_folder.mkdir(parents=True, exist_ok=True)
fig.write_html(plot_folder / f"notes_UMAP_2d_plot.html")
