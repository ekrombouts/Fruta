import datetime as dt
import os

import pandas as pd
import streamlit as st

# Volledige namen, zodat de app niet afhangt van de database waarin hij is aangemaakt
PROFILES_TABLE = "FRUTA.PUBLIC.PROFILES"
SCENARIOS_TABLE = "FRUTA.PUBLIC.SCENARIOS"
RECORDS_TABLE = "FRUTA.PUBLIC.RECORDS"

st.set_page_config(
    page_title="Fruta Data",
    page_icon=":banana:",
    layout="centered",
)

session = st.connection("snowflake", ttl=os.getenv("SNOWFLAKE_CONNECTION_TTL")).session()


@st.cache_data
def load_profiles():
    return session.table(PROFILES_TABLE).to_pandas()


@st.cache_data
def load_scenarios(client_id):
    return session.sql(
        f"SELECT * FROM {SCENARIOS_TABLE} WHERE CLIENT_ID = ?", params=[client_id]
    ).to_pandas()


@st.cache_data
def load_records(client_id):
    df = session.sql(
        f"SELECT * FROM {RECORDS_TABLE} WHERE CLIENT_ID = ? ORDER BY DATE",
        params=[client_id],
    ).to_pandas()
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
    return df


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


def select_date_range_sidebar(records_df):
    with st.sidebar:
        st.markdown("### Selecteer periode")
        start_date = st.date_input(
            "Startdatum:",
            value=records_df["DATE"].min() if not records_df.empty else dt.date.today(),
        )
        end_date = st.date_input(
            "Einddatum:",
            value=records_df["DATE"].max() if not records_df.empty else dt.date.today(),
        )
        return start_date, end_date


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


def display_scenarios(client_id):
    st.subheader("Scenario")
    scenarios_df = load_scenarios(client_id)
    if scenarios_df.empty:
        st.warning("Geen scenario's gevonden voor deze cliënt.")
        return

    scenarios_df = scenarios_df[["DATE_START_OF_WEEK", "EVENTS_DESCRIPTION"]].copy()
    scenarios_df["DATE_START_OF_WEEK"] = pd.to_datetime(
        scenarios_df["DATE_START_OF_WEEK"]
    ).dt.date
    scenarios_df = scenarios_df.rename(
        columns={"DATE_START_OF_WEEK": "Week", "EVENTS_DESCRIPTION": "Scenario"}
    )
    st.markdown(
        scenarios_df.to_html(index=False, justify="left"), unsafe_allow_html=True
    )


def display_reports(records_df, start_date, end_date):
    st.subheader("Rapportages")
    period_df = records_df[
        (records_df["DATE"] >= start_date) & (records_df["DATE"] <= end_date)
    ]
    if period_df.empty:
        st.info("Geen rapportages gevonden voor de opgegeven periode.")
        return

    for _, row in period_df.iterrows():
        st.markdown(f"- **{row['DATE']}**: {row['NOTE']}")


def display_info():
    st.subheader("Informatie")
    st.markdown(
        """
Deze app toont FrutaData, een fictieve zorgdatabase van verpleeghuis Fruta.
De volledig synthetische data bevatten cliëntprofielen, scenario's en automatisch gegenereerde zorgrapportages. Ze bieden een veilige testomgeving voor AI in de langdurige zorg.

De app en dataset vormen onderdeel van mijn leerproces. De code kan rommelig zijn en commentaar is niet altijd netjes, aanwezig of in de juiste taal.

[Bekijk de Fruta-collectie op Hugging Face](https://huggingface.co/collections/ekrombouts/fruta-680349c74db35a450da33f64)
"""
    )


st.title("🍎 Fruta Dashboard 🍎")

with st.spinner("Data laden..."):
    profiles_df = load_profiles()

client = select_client(profiles_df)
records_df = load_records(client["CLIENT_ID"])
start_date, end_date = select_date_range_sidebar(records_df)

tab1, tab2, tab3, tab4 = st.tabs(["Cliëntprofiel", "Scenario", "Rapportages", "Info"])

with tab1:
    display_client_profile(client)

with tab2:
    display_scenarios(client["CLIENT_ID"])

with tab3:
    display_reports(records_df, start_date, end_date)

with tab4:
    display_info()
