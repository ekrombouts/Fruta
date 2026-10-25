import datetime as dt

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from backend.database.fruta_queries import FrutaClient, FrutaClients
from backend.services.fall_risk_analyzer import FallRiskAnalyzer
from backend.services.generate_client_embedding_plot import create_client_embedding_plot
from dotenv import load_dotenv

load_dotenv()


def main():
    st.set_page_config(
        page_title="Fruta Data",
        page_icon=":banana:",
        layout="centered",
        initial_sidebar_state="auto",
    )
    st.title("🍎 Fruta Dashboard 🍎")

    with st.spinner("Loading data..."):
        clients_df = FrutaClients().clients
        st.markdown(
            "[Bekijk de Fruta-collectie op Hugging Face](https://huggingface.co/collections/ekrombouts/fruta-680349c74db35a450da33f64)"
        )

    selected_ward, client_id = select_client(clients_df)
    Fruta_client = FrutaClient(client_id)

    client_records = Fruta_client.get_notes()
    client_records["date"] = pd.to_datetime(client_records["date"]).dt.date

    selected_start_date, selected_end_date = select_date_range_sidebar(client_records)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Cliëntprofiel",
            "Scenario",
            "Rapportages",
            "Visualisatie",
            "Valrisico",
            "Info",
        ]
    )

    with tab1:
        display_client_profile(Fruta_client)

    with tab2:
        display_scenarios(Fruta_client)

    with tab3:
        display_reports(Fruta_client, selected_start_date, selected_end_date)

    with tab4:
        if st.button("📊 Toon embedding plot"):
            display_client_embedding_plot(client_id)

    with tab5:
        if st.button("⚠️ Start valrisico-analyse"):
            display_fall_risk(Fruta_client, selected_start_date, selected_end_date)

    with tab6:
        st.subheader("ℹ️ Informatie")
        st.markdown(
            f"""
            Deze app toont FrutaData, een fictieve zorgdatabase van verpleeghuis Fruta.
De volledig synthetische data bevatten cliëntprofielen, scenario’s en automatisch gegenereerde zorgrapportages. Ze bieden een veilige testomgeving voor AI in de langdurige zorg.

De app en dataset vormen onderdeel van mijn leerproces. De code kan rommelig zijn en commentaar is niet altijd netjes, aanwezig of in de juiste taal. 
"""
        )


def select_client(clients_df):
    col1, col2 = st.columns(2)
    with col1:
        selected_ward = st.selectbox(
            "Selecteer afdeling:", options=clients_df["ward"].unique()
        )
    with col2:
        client_id = st.selectbox(
            "Selecteer cliënt:",
            options=clients_df[clients_df["ward"] == selected_ward]["client_id"],
            format_func=lambda x: clients_df[clients_df["client_id"] == x][
                "achternaam"
            ].values[0],
        )
    return selected_ward, client_id


def display_client_profile(Fruta_client):
    st.subheader(f"🪪 Profiel van {Fruta_client.naam}")
    st.markdown(
        f"""
        | **Kenmerk**         | **Beschrijving**                      |
        |---------------------|---------------------------------|
        | **Afdeling**        | {Fruta_client.ward}       |
        | **Diagnose**        | {Fruta_client.diagnose} |
        | **Somatiek**        | {Fruta_client.somatiek}   |
        | **ADL**             | {Fruta_client.adl}        |
        | **Mobiliteit**      | {Fruta_client.mobiliteit}   |
        | **Gedrag**          | {Fruta_client.gedrag}   |
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        f"<small>**Cliënt ID:** {Fruta_client.client_id}  |  "
        f"**Model:** {Fruta_client.model}  |  "
        f"**Startdatum in zorg:** {Fruta_client.start_date}  |  "
        f"**Aantal weken in zorg:** {Fruta_client.duration}  |  "
        f"**Complicaties:** {Fruta_client.complications}</small>",
        unsafe_allow_html=True,
    )


def display_scenarios(Fruta_client):
    st.subheader("🎮 Scenario")
    client_scenarios = Fruta_client.get_scenario()
    if not client_scenarios.empty:
        scenarios_df = client_scenarios[
            ["date_start_of_week", "events_description"]
        ].copy()
        scenarios_df["date_start_of_week"] = pd.to_datetime(
            scenarios_df["date_start_of_week"]
        )
        scenarios_df["date_start_of_week"] = scenarios_df["date_start_of_week"].dt.date
        scenarios_df = scenarios_df.rename(
            columns={"date_start_of_week": "Week", "events_description": "Scenario"}
        )
        st.markdown(
            scenarios_df.to_html(index=False, justify="left"), unsafe_allow_html=True
        )
    else:
        st.warning("Geen scenario's gevonden voor deze cliënt.")


def display_reports(Fruta_client, selected_start_date, selected_end_date):
    st.subheader("📋 Rapportages")
    if selected_start_date and selected_end_date:
        st.markdown("#### Rapportages")
        client_notes = Fruta_client.get_notes(
            start_date=selected_start_date, end_date=selected_end_date
        )
        if not client_notes.empty:
            for _, row in client_notes.iterrows():
                st.markdown(f"- **{row['date']}**: {row['note']}")
        else:
            st.info("Geen rapportages gevonden voor de opgegeven periode.")


def select_date_range_sidebar(client_records):
    with st.sidebar:
        st.markdown("### Selecteer periode")
        start_date = st.date_input(
            "Startdatum:",
            value=(
                client_records["date"].min()
                if not client_records.empty
                else dt.date.today()
            ),
        )
        end_date = st.date_input(
            "Einddatum:",
            value=(
                client_records["date"].max()
                if not client_records.empty
                else dt.date.today()
            ),
        )
        return start_date, end_date


def display_client_embedding_plot(client_id):
    with st.spinner("Genereren van de embedding plot..."):
        html_path = create_client_embedding_plot(client_id)
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=600, width=700, scrolling=True)


def display_fall_risk(Fruta_client, selected_start_date, selected_end_date):
    if selected_start_date and selected_end_date:
        with st.spinner("Analyseren van valrisico..."):
            fra = FallRiskAnalyzer(
                client_id=Fruta_client.client_id,
                start_date=selected_start_date,
                end_date=selected_end_date,
                limit=10,
            )
            result = fra.analyze()

        # Belangrijkste uitkomst
        st.markdown(f"### Valrisico: {result.valrisico.value}")
        st.markdown(
            f"### Valincident: {'Ja' if result.valincident else 'Nee'} (Laatste incident: {result.datum_laatste_valincident})"
        )

        # Context tonen
        st.markdown("#### Relevante context:")
        st.dataframe(
            fra.context.sort_values(by="date")[["date", "content", "distance"]]
        )

        # Overwegingen tonen
        st.markdown("#### Overwegingen LLM:")
        for i, gedachte in enumerate(result.gedachtengang, 1):
            st.markdown(f"{i}. {gedachte}")


if __name__ == "__main__":
    main()
