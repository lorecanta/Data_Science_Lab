import pandas as pd
from helpers import utils_data_preparation

def process_financial_literacy_data(file_path: str) -> pd.DataFrame:
    """
    Carica, trasforma e calcola i punteggi sul dataset di literacy finanziaria.

    Questa funzione esegue i seguenti passaggi:
    1. Carica i dati da un file CSV.
    2. Applica la funzione di trasformazione per pulire e preparare i dati.
    3. Categorizza l'istruzione e la generazione per ciascun record.
    4. Crea una variabile "segmentation" combinando le categorie di istruzione e generazione.
    5. Calcola i punteggi per le variabili specifiche (conoscenza, comportamento, atteggiamento, ecc.).
    
    Parametri:
    - file_path (str): Il percorso al file CSV contenente i dati di literacy finanziaria.

    Ritorna:
    - pandas.DataFrame: Un DataFrame trasformato con colonne aggiuntive per "education", "generation", 
      "segmentation" e i punteggi calcolati.
    """
    # 1. Caricamento dei dati dal file CSV
    df = pd.read_csv(file_path)
    
    # 2. Trasformazione dei dati per la pulizia e la codifica
    df_transformed = utils_data_preparation.transform_dataframe(df)
    
    # 3. Categorizzazione dell'istruzione (education) e della generazione (generation)
    df_transformed["education"] = df_transformed["qd9"].apply(utils_data_preparation.categorize_education)
    df_transformed["generation"] = df_transformed["qd7"].apply(utils_data_preparation.categorize_generation)
    
    # 4. Creazione della variabile 'segmentation' come combinazione di 'education' e 'generation'
    df_transformed["segmentation"] = df_transformed["education"].astype(str) + "_" + df_transformed["generation"].astype(str)
    
    # 5. Calcolo dei punteggi per le variabili (conoscenza, comportamento, atteggiamento, ecc.)
    df_transformed = utils_data_preparation.calculate_scores(df_transformed)
    
    return df_transformed
