
# Data Science Lab Project: Financial Literacy Analysis (2017-2020)

## Descrizione del progetto

Questo progetto si concentra sull'analisi dei dati di alfabetizzazione finanziaria raccolti dalla Banca d'Italia nel 2017 e nel 2020. Il dataset contiene informazioni riguardo le conoscenze, le attitudini e i comportamenti finanziari dei cittadini italiani. L'obiettivo del progetto è eseguire una serie di trasformazioni sui dati e analizzare l'evoluzione di questi aspetti nel corso di tre anni, utilizzando metodi di analisi delle regole di associazione basati sulla *Formal Concept Analysis*.

### Struttura del progetto

Il progetto è organizzato nelle seguenti cartelle:

- **`data/`**: Contiene i file CSV con i dati relativi all'alfabetizzazione finanziaria.
    - `Financia_literacy_2017.csv`: Dati del sondaggio del 2017.
    - `Financia_literacy_2020.csv`: Dati del sondaggio del 2020.
  
- **`shared/`**: Contiene due moduli Python (utils) che gestiscono la preparazione dei dati e l'analisi.
    - `utils_data_preparation.py`: Contiene funzioni per il caricamento, la trasformazione e la pulizia dei dati.
    - `utils_analysis.py`: Contiene funzioni per l'analisi dei dati, inclusa l'analisi delle regole di associazione.
  
- **`main.ipynb`**: Notebook principale che esegue l'intero flusso di lavoro, includendo il caricamento dei dati, la preparazione e la trasformazione, l'analisi delle regole di associazione e la visualizzazione dei risultati.

---

## Requisiti

Questo progetto richiede che tu abbia installato tutte le dipendenze elencate nel file `requirements.txt`. Per installarle, puoi eseguire il seguente comando:

```bash
pip install -r requirements.txt
```

---

## Come utilizzare il progetto

### Esegui l'intero flusso di lavoro

1. **Installa i requisiti**: Esegui il comando `pip install -r requirements.txt` per installare tutte le dipendenze necessarie.

2. **Esegui il notebook**: Una volta che le dipendenze sono state installate, apri e esegui il notebook `main.ipynb`. Il notebook caricherà i dati, li trasformerà, eseguirà l'analisi delle regole di associazione e visualizzerà i risultati: 
    - **Carica i dati**: I dati sono caricati dai file CSV `Financia_literacy_2017.csv` e `Financia_literacy_2020.csv` situati nella cartella `data/`.
    - **Preprocessing dei dati**: Utilizza il modulo `utils_data_preparation` per preparare e trasformare i dati, creando nuove variabili come `education`, `generation`, e `segmentation`.
    - **Analisi delle regole di associazione**: Il cuore dell'analisi si basa sulla classe `AssociationRules`, che implementa i metodi per calcolare le metriche di *support*, *confidence* e *lift* tra variabili, utilizzando la *Formal Concept Analysis*.

    La classe `AssociationRules` ha i seguenti metodi principali:
    - **`support(itemset)`**: Calcola il supporto di un insieme di item.
    - **`confidence(itemset_A, itemset_B)`**: Calcola la confidenza della regola A -> B.
    - **`lift(itemset_A, itemset_B)`**: Calcola il lift della regola A -> B.
    - **`get_combinations(itemset)`**: Genera tutte le combinazioni di sottoinsiemi per un dato insieme.
    - **`calculate_all_metrics_for_selected_sets()`**: Calcola supporto, confidenza e lift per tutte le combinazioni di variabili.
    - **`filter_by_values(df, min_support, min_confidence, min_lift)`**: Filtra le regole di associazione in base ai valori minimi di supporto, confidenza e lift.

---

## Dettagli delle funzioni principali

- **`AssociationRules`**: Classe che calcola le metriche di supporto, confidenza e lift per le regole di associazione tra variabili A e B. Utilizza la *Formal Concept Analysis* per generare e analizzare le combinazioni di variabili.
  
  - **`support(itemset)`**: Calcola il supporto di un insieme di item.
  - **`confidence(itemset_A, itemset_B)`**: Calcola la confidenza della regola A -> B.
  - **`lift(itemset_A, itemset_B)`**: Calcola il lift della regola A -> B.
  - **`get_combinations(itemset)`**: Genera tutte le combinazioni di sottoinsiemi per un dato insieme.
  - **`calculate_all_metrics_for_selected_sets()`**: Calcola Supporto, Confidenza e Lift per le combinazioni selezionate di A e B.
  - **`filter_by_values(df, min_support, min_confidence, min_lift)`**: Filtra le regole di associazione in base ai valori fissi per Support, Confidence e Lift.

- **`utils_data_preparation`**: Contiene le funzioni di preparazione dei dati, tra cui la trasformazione dei dati e la categorizzazione in base a variabili come `education` e `generation`.

- **`utils_analysis`**: Contiene funzioni di analisi delle regole di associazione, tra cui la visualizzazione dei risultati e il calcolo delle metriche di supporto, confidenza e lift.

---

## Risultati

I risultati delle analisi (inclusi i grafici e le statistiche descrittive) verranno esportati nella cartella `output/`. 

