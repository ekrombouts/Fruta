import ast
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.io as pio
from backend.database.db_connection import Database
from backend.llm.embedding_visualizer import plot_client_embeddings
from sqlalchemy import text

# Paths to required models and output files
root = Path(__file__).resolve().parents[2]
umap_model_path = root / "backend" / "models" / "notes_umap_2d.pkl"
topic_map_path = root / "backend" / "static" / "topic_map.png"
client_plot_html_path = root / "backend" / "static" / "output" / "client_plot.html"


def create_client_embedding_plot(client_id: str):
    """
    Prepares a 2D embedding plot for a specific client based on their notes and embeddings.

    Args:
        client_id (str): The ID of the client.

    Returns:
        str: Path to the generated HTML file containing the plot.
    """
    # Initialize database connection
    engine = Database().get_engine()

    # SQL query to fetch client records and embeddings
    query = f"""SELECT r.note_id, r.client_id, r.note, r.nfi_embedding as embedding, r.date, c.achternaam
FROM records r
LEFT JOIN profiles c ON r.client_id = c.client_id
WHERE r.client_id = '{client_id}'
ORDER BY r.date;
"""

    # Execute the query and load results into a DataFrame
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Load the pre-trained UMAP model for dimensionality reduction
    reducer = joblib.load(umap_model_path)

    # Helper function to safely parse embeddings from strings
    def safe_parse_embedding(x):
        if isinstance(x, str):
            return np.array(ast.literal_eval(x), dtype=np.float32)
        return np.array(x, dtype=np.float32)

    # The embeddings are parsed into NumPy arrays, then stacked into a 2D array. This format is required by the UMAP model to perform dimensionality reduction.
    df["embedding"] = df["embedding"].apply(safe_parse_embedding)
    embeddings = np.vstack(df["embedding"])

    # Reduce embeddings to 2D using the UMAP model
    reduced_embeddings = reducer.transform(embeddings)

    # Create a DataFrame for plotting
    df_plot = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
    df_plot["text"] = df["note"]  # Add note text for hover information
    df_plot["date"] = pd.to_datetime(df["date"])  # Convert datetime column
    df_plot["time_diff"] = (
        df_plot["date"] - df_plot["date"].min()
    ).dt.days  # Calculate time difference
    df_plot["achternaam"] = df["achternaam"].iloc[0]  # Add client name

    # Generate the plot using the embedding visualizer
    fig = plot_client_embeddings(df_plot, topic_map_path)

    # Save the plot as an HTML file
    pio.write_html(fig, file=client_plot_html_path, auto_open=False)

    return client_plot_html_path


if __name__ == "__main__":
    # Example usage: Generate a plot for a specific client
    client_id = "ca_01"
    html_file = create_client_embedding_plot(client_id)
    print(f"Plot saved to: {html_file}")
