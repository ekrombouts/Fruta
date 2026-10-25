# Verpleeghuis Fruta

Deze app toont de zorgdatabase van het fictieve verpleeghuis Fruta.  
De volledig synthetische data omvat cliëntprofielen, scenario's en automatisch gegenereerde zorgrapportages, en biedt een veilige testomgeving voor AI in de langdurige zorg.

Zowel de app als de dataset maken deel uit van mijn leerproces. De code kan rommelig zijn, met inconsistent of ontbrekend commentaar. En af en toe zal je ChatGPT herkennen ;-)

Voel je vrij om code en ideeën te gebruiken. Ik zou het wel leuk vinden om iets van je te horen:
- [LinkedIn](https://www.linkedin.com/in/evarombouts/)
- [Hugging Face](https://huggingface.co/ekrombouts)
- [GitHub](https://github.com/ekrombouts)

## Setup en installatie

### PostgreSQL Docker Setup
De PostgreSQL-database kan draaien in een Docker-container, maar lokaal installeren kan ook.
Onderstaande stappen beschrijven de Docker-variant om snel aan de slag te gaan met een kant-en-klare setup. Uiteraard moet Docker hiervoor zijn geinstalleerd op je systeem. 

PostgreSQL wordt hier geconfigureerd als een eenvoudige vector database, geschikt voor similarity search met embeddings. Hiervoor wordt de `vector` extensie gebruikt. Deze wordt automatisch geactiveerd in de code, maar let op: Als je een eigen database-installatie gebruikt kunnen er extra stappen nodig zijn. Voor meer informatie over het gebruik van PostgreSQL als vector database met pgvector, kun je [deze](https://www.timescale.com/blog/postgresql-as-a-vector-database-create-store-and-query-openai-embeddings-with-pgvector) tutorial raadplegen.

#### 1. Voorbereiding

Maak een gedeeld netwerk aan (eenmalig):

```bash
docker network create fruta-net
```

Dit zorgt ervoor dat eventuele toekomstige containers kunnen communiceren met de database.

#### 2. Instellen databasewachtwoord

Kopieer het voorbeeldbestand:

```bash
cp .envexample .env
```

Pas in .env de waarde van DB_PASSWORD aan naar een zelfgekozen wachtwoord.
Docker Compose leest deze variabele automatisch in (zorg wel dat ze beide in de root-folder blijven staan).

#### 3. Start de Database

Start de container:

```bash
docker compose -f docker-compose.db.yml -p fruta-db up -d
```
Dit bouwt een image (als die nog niet bestaat) en start een container met:
- PostgreSQL als database
- Een persistent volume voor opslag (fruta-db_fruta_data)
- Een extern netwerk (fruta-net) voor koppeling met andere containers

### Omgeving instellen

#### 1. Virtual environment
Maak een venv aan, activeer en installeer de requirements:

Voor macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Voor Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Pythonpath
Om imports uit de app-map goed te laten werken, stel je het PYTHONPATH in.

Zelf doe ik dit via de .code-workspace file (zie het voorbeeld), maar je kunt het ook handmatig doen. Dit moet je dan bij elke nieuwe terminalsessie opnieuw instellen.

macOS/Linux (terminal):
```bash
export PYTHONPATH="$(pwd)/app"
```

Windows (PowerShell):
```powershell
$env:PYTHONPATH="$PWD\app"
```

Zo voorkom je importfouten.

### Database vullen

#### 1. Test de databaseverbinding

Controleer of de verbinding werkt met:

```bash
python app/backend/database/db_connection.py
```

#### 2. Vul de Database

Vul de database met de Hugging Face data via:

```bash
python scripts/01_populate_database.py
```

#### 3. Embeddings genereren

We gaan nu de rapportages omzetten naar embeddings. 
Controleer hiervoor eerst de settings in de config files ```app/backend/config/settings.py``` en ```app/backend/config/llm_config.py```.

Controleer of de embedding factory werkt en de instellnigen juist staan: 

```bash
python app/backend/llm/embedding_factory.py
```

en draai vervolgens dit script:
```bash
python scripts/02_upsert_embeddings.py
```

De embeddings vormen de basis voor onze verdere toepassingen. 

### Trainen UMAP model

Gebruik het volgende script om het UMAP-model te trainen:

```bash
python scripts/03_reduce_embeddings.py
```
UMAP reduceert de hoog-dimensionale embeddings naar twee dimensies, zodat ze gevisualiseerd kunnen worden in een interactieve plot.
Het getrainde model wordt opgeslagen als ```app/backend/models/notes_umap_2d.pkl``` en wordt later gebruikt in de app.

De visualisatie wordt opgeslagen als html file. Daarnaast heb ik herkenbare onderwerpen getekend op een soort “landkaart”, bedoeld als achtergrond bij cliëntvisualisaties in de app. De kaart is met de hand getekend, maar zou ongeveer moeten overeenkomen met wat het script oplevert. 

Open de bestanden in je browser en controleer of ze globaal overeen komen:
```bash
# Mac
open output/notes_UMAP_2d_plot.html
open app/backend/static/topic_map.png

# Linux
xdg-open output/notes_UMAP_2d_plot.html
xdg-open app/backend/static/topic_map.png

# Windows (PowerShell)
start output/notes_UMAP_2d_plot.html
start app/backend/static/topic_map.png

```

