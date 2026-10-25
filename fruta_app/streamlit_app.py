import os

import streamlit as st

# Volledige namen, zodat de app niet afhangt van de database waarin hij is aangemaakt
PROFILES_TABLE = "FRUTA.PUBLIC.PROFILES"

st.set_page_config(
    page_title="Fruta Data",
    page_icon=":banana:",
    layout="centered",
)

session = st.connection("snowflake", ttl=os.getenv("SNOWFLAKE_CONNECTION_TTL")).session()


@st.cache_data
def load_profiles():
    return session.table(PROFILES_TABLE).to_pandas()


def format_naam(client):
    titel = "Mevrouw" if client["GESLACHT"] == "v" else "Meneer"
    return f"{titel} {client['VOORNAAM']} {client['ACHTERNAAM']}"


def select_client(profiles_df):
    col1, col2 = st.columns(2)
    with col1:
        selected_ward = st.selectbox(
            "Selecteer afdeling:", options=profiles_df["WARD"].unique()
        )
    with col2:
        ward_df = profiles_df[profiles_df["WARD"] == selected_ward]
        client_id = st.selectbox(
            "Selecteer cliënt:",
            options=ward_df["CLIENT_ID"],
            format_func=lambda x: ward_df[ward_df["CLIENT_ID"] == x]["ACHTERNAAM"].values[0],
        )
    return profiles_df[profiles_df["CLIENT_ID"] == client_id].iloc[0]


def display_client_profile(client):
    st.subheader(f"Profiel van {format_naam(client)}")
    st.markdown(
        f"""
        | **Kenmerk**         | **Beschrijving**                      |
        |---------------------|---------------------------------|
        | **Afdeling**        | {client['WARD']}       |
        | **Diagnose**        | {client['DIAGNOSE']} |
        | **Somatiek**        | {client['SOMATIEK']}   |
        | **ADL**             | {client['ADL']}        |
        | **Mobiliteit**      | {client['MOBILITEIT']}   |
        | **Gedrag**          | {client['GEDRAG']}   |
        """
    )
    st.markdown("---")
    st.caption(
        "Over de synthetische data: met welk model de rapportages en scenario's van deze cliënt "
        "zijn gegenereerd, en de instellingen van het scenario."
    )
    st.markdown(
        f"<small>**Cliënt ID:** {client['CLIENT_ID']}  |  "
        f"**Model:** {client['MODEL']}  |  "
        f"**Startdatum in zorg:** {client['START_DATE']}  |  "
        f"**Aantal weken in zorg:** {client['DURATION']}  |  "
        f"**Complicaties:** {client['COMPLICATIONS']}</small>",
        unsafe_allow_html=True,
    )


st.title("🍎 Fruta Dashboard 🍎")

with st.spinner("Data laden..."):
    profiles_df = load_profiles()

client = select_client(profiles_df)

(tab1,) = st.tabs(["Cliëntprofiel"])

with tab1:
    display_client_profile(client)
