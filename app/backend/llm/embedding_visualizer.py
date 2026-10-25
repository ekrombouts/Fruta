import base64

import numpy as np
import pandas as pd
import plotly.express as px


def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return "data:image/png;base64," + base64.b64encode(img_file.read()).decode()


def create_interactive_plot(
    data, embedding_column, color_column, hover_column, title, caption
):
    """Genereert een interactieve 2D of 3D scatterplot van gereduceerde embeddings.

    Args:
        data (pd.DataFrame): DataFrame met de embeddings en metadata.
        embedding_column (str): Naam van de kolom met gereduceerde embeddings.
        color_column (str): Naam van de kolom voor kleurgroepering.
        hover_column (str): Naam van de kolom voor hover-informatie.
        title (str): Titel van de plot.
        caption (str): Onderschrift van de plot.

    Returns:
        plotly.graph_objects.Figure: De interactieve scatterplot.
    """

    data.reset_index(drop=True, inplace=True)
    reduced_embeddings = np.vstack(data[embedding_column])
    n_dim = reduced_embeddings.shape[1]

    if n_dim not in [2, 3]:
        raise ValueError("Reduced embeddings must have 2 or 3 dimensions for plotting.")

    columns = ["x", "y"] if n_dim == 2 else ["x", "y", "z"]
    df_plot = pd.DataFrame(reduced_embeddings, columns=columns)
    df_plot["hover_info"] = data[hover_column]
    df_plot["color"] = data[color_column]

    if n_dim == 2:
        fig = px.scatter(
            df_plot,
            x="x",
            y="y",
            hover_data=["hover_info"],
            color="color",
            title=title,
            labels={color_column: "Category"},
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
    else:
        fig = px.scatter_3d(
            df_plot,
            x="x",
            y="y",
            z="z",
            hover_data=["hover_info"],
            color="color",
            title=title,
            labels={color_column: "Category"},
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )

    fig.update_layout(
        annotations=[
            dict(
                text=caption,
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0,
                y=-0.1,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=12),
            )
        ]
    )
    fig.update_traces(marker=dict(size=5, line=dict(width=1, color="DarkSlateGrey")))
    return fig


def plot_client_embeddings(df_plot, background_image_path):
    """Plot the embeddings with animation and background image."""
    background_image_base64 = encode_image(background_image_path)

    # Copy all rows to the minimum time_diff
    min_time_diff = df_plot["time_diff"].max()
    df_copy = df_plot.copy()
    df_copy["time_diff"] = min_time_diff
    df_plot = pd.concat([df_plot, df_copy], ignore_index=True)

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        hover_data={"text": True, "date": True},
        title=f"Rapportages van {df_plot['achternaam'].iloc[0]}",
        animation_frame="time_diff",
        animation_group="text",
        labels={"time_diff": "Time (days)"},
    )

    # Voeg achtergrondafbeelding toe
    fig.update_layout(
        images=[
            dict(
                source=str(background_image_base64),
                xref="x",
                yref="y",
                x=-6,
                y=-4,
                sizex=18,
                sizey=18,
                xanchor="left",
                yanchor="bottom",
                layer="below",
            )
        ]
    )

    # Configureer de assen
    fig.update_xaxes(
        range=[-6, 12],
        dtick=2,
        showline=True,
        linewidth=2,
        linecolor="#103C6D",
        mirror=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="#96C1CA",
    )
    fig.update_yaxes(
        range=[-4, 14],
        dtick=2,
        showline=True,
        linewidth=2,
        linecolor="#103C6D",
        mirror=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="#96C1CA",
    )

    fig.update_layout(
        height=600,
        width=665,
        font=dict(family="Arial, sans-serif", size=14, color="#103C6D"),
        plot_bgcolor="#FFF5ED",
        paper_bgcolor="white",
    )

    # Pas de animatiesnelheid aan
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 50

    return fig
