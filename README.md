# Verpleeghuis Fruta

Dit project toont de zorgdatabase van het fictieve verpleeghuis Fruta, als Streamlit-app in Snowflake.
De volledig synthetische data omvat cliëntprofielen, scenario's en automatisch gegenereerde zorgrapportages, en biedt een veilige testomgeving voor AI in de langdurige zorg.

Zowel de app als de dataset maken deel uit van mijn leerproces. De code kan rommelig zijn, met inconsistent of ontbrekend commentaar. En af en toe zal je ChatGPT herkennen ;-)

Voel je vrij om code en ideeën te gebruiken. Ik zou het wel leuk vinden om iets van je te horen:
- [LinkedIn](https://www.linkedin.com/in/evarombouts/)
- [Hugging Face](https://huggingface.co/ekrombouts)
- [GitHub](https://github.com/ekrombouts)

## Wat zit erin

| Map / bestand | Wat |
|---|---|
| `fruta_app/` | De Streamlit-app voor Streamlit in Snowflake, met de tabs Cliëntprofiel, Scenario, Rapportages, Visualisatie en Info |
| `notebooks/01_populate_snowflake.ipynb` | Laadt de [Fruta-datasets van Hugging Face](https://huggingface.co/collections/ekrombouts/fruta-680349c74db35a450da33f64) in Snowflake (lokaal) |
| `notebooks/02_embeddings_local.ipynb` | Maakt embeddings van de rapportages met `snowflake-arctic-embed-l-v2.0` (lokaal) |
| `notebooks/03_reduce_embeddings.ipynb` | Traint UMAP en zet de rapportages op een 2D-kaart voor de visualisatie (in Snowflake) |
| `notebooks/1X_*.ipynb` | Experiment: het embeddingmodel aanpassen aan de taal van zorgrapportages (zie verderop) |
| `notebooks/snowflake_setup.sql` | Eenmalige setup van database en warehouse |

## Setup

### Vereisten
- Een Snowflake-account. Een trialaccount werkt, met beperkingen (zie verderop).
- [uv](https://docs.astral.sh/uv/) voor de lokale Python-omgeving.

### 1. Snowflake voorbereiden
Voer `notebooks/snowflake_setup.sql` uit in een SQL-worksheet in Snowsight. Dit maakt de database `FRUTA` en het warehouse `COMPUTE_WH` aan.

### 2. Lokale omgeving
```bash
uv sync
```
Dit maakt een `.venv` aan met de juiste Python-versie en alle pakketten. Kies die `.venv` als kernel voor de notebooks.

Maak in de root een `.env` met je Snowflake-gegevens:
```
SNOWFLAKE_ACCOUNT=...
SNOWFLAKE_USER=...
SNOWFLAKE_PASSWORD=...
```

### 3. Data laden en voorbereiden
Draai de notebooks op volgorde:

1. **`01_populate_snowflake.ipynb`** (lokaal): schrijft de tabellen `PROFILES`, `SCENARIOS`, `RECORDS` en `NOTES` naar `FRUTA.PUBLIC`.
2. **`02_embeddings_local.ipynb`** (lokaal): maakt `RECORD_EMBEDDINGS` en `NOTE_EMBEDDINGS`. Dit gebeurt lokaal, omdat `AI_EMBED` niet beschikbaar is op een trialaccount. Bij opnieuw draaien worden alleen ontbrekende embeddings gemaakt.
3. **`03_reduce_embeddings.ipynb`** (in Snowflake): traint UMAP op de `NOTES` en zet ook de `RECORDS` op dezelfde kaart. Dit maakt `NOTE_UMAP_2D` en `RECORD_UMAP_2D`, en bewaart het model en een plot per categorie op de stage `@FRUTA.PUBLIC.MODELS`. Vereist `umap-learn`, `matplotlib` en `plotly`.

### 4. De app
Deploy `fruta_app/` als Streamlit-app in Snowflake, bijvoorbeeld vanuit een Workspace die aan deze repository gekoppeld is. `fruta_app/snowflake.yml` beschrijft de app (warehouse `COMPUTE_WH`, compute pool `SYSTEM_COMPUTE_POOL_CPU`).

De app leest `PROFILES`, `SCENARIOS`, `RECORDS`, `RECORD_UMAP_2D` en `NOTE_UMAP_2D`. De tab Visualisatie toont de rapportages van een cliënt op de 2D-kaart, met de `NOTES` per onderwerp als vage achtergrond.

## Beperkingen van een trialaccount
Cortex AI-functies zoals `AI_COMPLETE` en `AI_EMBED` werken niet op een trialaccount. Daarom:
- worden de embeddings lokaal gemaakt (`02_embeddings_local.ipynb`);
- staan de tabs **Valrisico** en **Vraag** (nog) niet in de app. Die gebruikten een LLM. De eerdere versie van de app (PostgreSQL + OpenAI), met deze functies, staat in de git-geschiedenis.

## Experiment: embeddings die de zorgtaal beter verstaan
Het plan staat bovenaan `notebooks/10_meetlat_embeddings.ipynb`. In het kort: het embeddingmodel verder trainen op zorgrapportages, met MLM en TSDAE, zodat het jargon en schrijfwijzen beter herkent. Elk model wordt met dezelfde meetlat gemeten. De notebooks zijn genummerd als `1X`, waarbij X de stap in het plan is.

- `10_meetlat_embeddings.ipynb`: meetlat en nulmeting, met de resultaten in `EMBEDDING_EVAL` (lokaal)
- `13_tsdae.ipynb`: TSDAE-training (lokaal). Het model komt in `models/`, buiten git.
