import json

import pandas as pd
from backend.config.settings import get_settings
from backend.llm.llm_factory import LLMService
from pydantic import BaseModel, Field


class Problem(BaseModel):
    probleem: str = Field(..., description="Korte naam van het probleem")
    beschrijving: str = Field(
        ..., description="Uitgebreide beschrijving van het probleem"
    )


class Answer(BaseModel):
    Problemen: list[Problem]


query = """
Hieronder staan rapportages van een client in het verpleeghuis. Jouw taak is om de drie belangrijkste problemen van de client te identificeren, om te gebruiken in een zorgplan.
Zorg dat de problemen algemeen geformuleerd zijn, passend bij problemen zoals we die zien in het verpleeghuis.
Niet op een specifiek detail.
"""

context = """CONTEXT
Ochtendverzorging uitgevoerd. Dhr. maakte gefrustreerde indruk. Gaf via computer aan zich opgesloten te voelen. Extra tijd genomen voor gesprek en geruststelling.
Bezoek van familie. Dhr. knapte zichtbaar op. Hebben samen naar fotoalbum gekeken. Na bezoek was stemming merkbaar verbeterd.
Avondzorg verleend. Dhr. gaf aan moe maar tevreden te zijn na familiebezoek. Wisselligging toegepast en extra kussen voor ondersteuning gebruikt.
Ochtendzorg gedaan. Dhr. had redelijk geslapen. Huidcontrole: lichte roodheid op stuit opgemerkt. Extra aandacht besteed aan positionering.
Fysiotherapeut geweest voor passieve bewegingsoefeningen. Dhr. gaf aan dit als prettig te ervaren. Nadien sondevoeding gegeven.
Avondronde. Dhr. klaagde over jeuk aan neus. Kon zelf niet krabben. Geholpen en extra aandacht besteed aan gezichtsverzorging.
Vroege dienst. Dhr. al wakker bij binnenkomst. Gaf aan slecht geslapen te hebben door onrustige gedachten. Verzorging gedaan en extra tijd genomen voor praatje.
Logopedist terug voor vervolg slikonderzoek. Dhr. erg vermoeid, onderzoek vroegtijdig gestopt. Rust ingelast.
Late avondronde. Dhr. sliep al bij binnenkomst. Wisselligging toegepast zonder wakker te maken. Ademhaling rustig en regelmatig.
Ochtendzorg later uitgevoerd op verzoek van dhr. Gaf aan eindelijk goed geslapen te hebben. ADL-zorg verleend, geen bijzonderheden.
Ergotherapeut langs geweest voor aanpassing communicatieapparaat. Dhr. kan nu sneller en makkelijker communiceren. Was hier erg blij mee.
Avondzorg verricht. Dhr. vroeg om muziek. Via oogbesturing playlist samengesteld. Leek ontspannen in te slapen met rustgevende muziek op achtergrond.
Dhr. wakker gemaakt voor ochtendverzorging. Ogen geopend na lichte aanraking. ADL volledig overgenomen. Huid ziet er goed uit, geen roodheid gezien. Oogbesturingscomputer aangezet zodat dhr. kan communiceren. Gaf aan goed geslapen te hebben.
"""

prompt = query + context

# 2. Bouw de prompt
messages = [
    {
        "role": "system",
        "content": "Je bent gespecialiseerd in het samenvatten van zorgdossiers",
    },
    {"role": "user", "content": prompt},
]

# 3. Haal de provider uit settings
provider_name = get_settings().llm_provider

# 4. Maak de LLM service
# llm = LLMService(provider=provider_name)
llm = LLMService(provider="ollama")

# 5. Genereer het antwoord
response_model, raw = llm.create_completion(
    response_model=Answer, messages=messages, model="llama3.2:1b"
)

# 6. Print het resultaat
print("Antwoord:", response_model)

print(json.dumps(response_model.dict(), ensure_ascii=False, indent=2))
