# Simulatore Indici Comuni 2025

App Streamlit per simulare, calcolare e confrontare gli indici della **Mappa dei Comuni Digitali 2025** a partire da un questionario dinamico generato da configurazione JSON.

L'app consente di:

- compilare un questionario simulato per un Comune
- calcolare gli indici di **livello 1, livello 2 e livello 3**
- personalizzare i pesi degli indici compositi
- confrontare il profilo di più Comuni simulati
- visualizzare ranking e radar plot tra Comuni

---

## Funzionalità principali

- **Questionario dinamico** costruito automaticamente dalla configurazione degli indici
- **Calcolo multilivello degli indici**:
  - Livello 1 (base)
  - Livello 2 (compositi)
  - Livello 3 (macro)
- **Override dei pesi** per gli indici di livello 2 e 3
- **Archivio simulato** dei Comuni creati nella sessione
- **Confronto tra Comuni** tramite radar plot
- **Debug tab** con dettaglio delle risposte e dello scoring

---

## Struttura del progetto

```bash
simulatore-indici-comuni-2025/
├── app_test.py
├── create_index.py
├── index_config_D2_D3.json
├── pyproject.toml
├── poetry.lock
├── README.md
└── .gitignore
```

---

## Requisiti

- Python **3.11** o **3.12**
- [Poetry](https://python-poetry.org/) per la gestione dell'environment e delle dipendenze

---

## Installazione

Clona la repository:

```bash
git clone <URL_DELLA_REPOSITORY>
cd simulatore-indici-comuni-2025
```

Configura Poetry per creare l'ambiente virtuale nella cartella del progetto:

```bash
poetry config virtualenvs.in-project true
```

Installa le dipendenze:

```bash
poetry install
```

Se il progetto è configurato in modalità non-package, Poetry verrà usato solo come dependency manager.

---

## Avvio dell'app

Per avviare l'app Streamlit:

```bash
poetry run streamlit run app.py
```

Una volta avviata, l'app sarà disponibile in locale all'indirizzo mostrato da Streamlit, tipicamente:

```bash
http://localhost:8501
```

---

## Configurazione

L'app legge una configurazione JSON contenente la definizione degli indici e delle regole di scoring.

File di configurazione atteso:

```bash
index_config_D2_D3.json
```

---

## Dipendenze principali

- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`

Dipendenze di sviluppo:

- `black`
- `ruff`
- `pytest`

---

## Come funziona

1. Inserisci il **nome del Comune**
2. Seleziona la **classe dimensionale**
3. Scegli il **livello** e l'**indice** da visualizzare
4. Compila il **questionario dinamico**
5. Premi **Calcola / aggiorna Comune simulato**
6. Esplora i risultati nelle tab:
   - **Indici calcolati**
   - **Profilo Comune**
   - **Ranking simulato**
   - **Debug**

---

## Output dell'app

L'app produce:

- dataframe con gli **indici di livello 1**
- dataframe con gli **indici di livello 2**
- dataframe con gli **indici di livello 3**
- confronto del Comune rispetto agli altri Comuni simulati
- ranking globale e per classe dimensionale
- visualizzazione delle componenti dell'indice selezionato

---

## Note tecniche

- I Comuni simulati vengono salvati in `st.session_state`, quindi l'archivio è valido solo nella sessione corrente.
- Il questionario viene generato a partire dalla configurazione degli indici di livello 1.
- I pesi modificati da sidebar vengono applicati in override rispetto alla configurazione originale.
- Il confronto radar è disponibile solo se nella sessione esistono almeno due Comuni simulati.

---

## Licenza

Questo progetto è distribuito sotto licenza **GNU Affero General Public License v3.0 (AGPL-3.0)**.

Puoi usare, studiare, modificare e ridistribuire il software secondo i termini della licenza AGPL-3.0.

Se il software viene modificato e reso disponibile attraverso una rete, il codice sorgente della versione modificata deve essere reso disponibile secondo i termini della stessa licenza.

Consulta il file `LICENSE` per il testo completo della licenza.

---

## Stato del progetto

Prototipo applicativo sviluppato per la simulazione e l'analisi degli indici comunali digitali 2025.

---

## Autore

**Gabbo**

---

## Possibili sviluppi futuri

- persistenza dell'archivio simulato su file o database
- esportazione dei risultati in CSV/Excel