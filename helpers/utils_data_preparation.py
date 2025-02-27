import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applica la trasformazione ai dati del DataFrame per la pulizia e la codifica.

    La funzione esegue diverse operazioni sul DataFrame, tra cui la gestione dei valori mancanti,
    la sostituzione di valori specifici con il valore di moda, e l'applicazione di una codifica one-hot
    per le variabili categoriche. Inoltre, stampa un messaggio che indica la presenza di eventuali valori mancanti.

    Parametri:
    - df (pandas.DataFrame): Il DataFrame contenente i dati da trasformare. Deve includere colonne numeriche e categoriche.

    Ritorna:
    - pandas.DataFrame: Un DataFrame trasformato con le variabili one-hot codificate e senza valori mancanti.
    """
    # Gestione dei valori mancanti
    df.fillna({"qprod1_d": 99, "qprod2": -99}, inplace=True)

    # Modifica le colonne 'qk4' e 'qk5' in base a determinate condizioni
    df['qk4'] = np.where(df['qk4'].isin([-97, -99, 0]), df['qk4'], 1)
    df['qk5'] = np.where(df['qk5'].isin([-97, -99, 102]), df['qk5'], 1)

    # Gestione delle colonne 'qf10_' sostituendo -97 e -99 con la moda
    qf10_cols = [col for col in df.columns if col.startswith("qf10_")]
    for col in qf10_cols:
        mode_value = df[col].mode()[0]
        df[col] = df[col].replace([-97, -99], mode_value)

    # Definizione delle variabili
    binary_variables = [
        "qd1", "SM", "qd12", "qprod1c_1", "qprod1c_2", "qprod1c_3", "qprod1c_5", "qprod1c_6", "qprod1c_7",
        "qprod1c_8", "qprod1c_10", "qprod1c_11", "qprod1c_12", "qprod1c_14", "qprod1c_99", "qf3_1", "qf3_3", "qf3_4",
        "qf3_6", "qf3_7", "qf3_8", "qf3_99", "qf9_1", "qf9_10", "qf9_2", "qf9_3", "qf9_4", "qf9_5", "qf9_6", "qf9_7",
        "qf9_8", "qf9_9", "qf9_99", "qprod3_1", "qprod3_2", "qprod3_3", "qprod3_4", "qprod3_5", "qprod3_6", "qprod3_7",
        "qprod3_8", "qprod3_9", "qprod3_10", "qprod3_11", "qprod3_12", "qprod3_13", "qprod3_14", "qprod3_15", "qprod3_16",
        "qprod3_17", "qprod3_18", "qprod3_99", "qf12_1_a", "qf12_1_b", "qf12_1_c", "qf12_2_d", "qf12_3_e", "qf12_3_f",
        "qf12_3_g", "qf12_4_k", "qf12_4_l", "qf12_5_m", "qf12_5_o", "qf12_6_p", "qf12_6_q", "qf12_7_r", "qf12_97", "qf12_99"
    ]
    integer_variables = ["qd5b", "qd7"] + qf10_cols
    continuous_variables = ["pesofitc"]

    # Variabili geodemografiche
    geodemo = ["AREA5", "qd10", "qd9"]
    
    # Variabili categoriche
    categorical_variables = df.columns.difference(binary_variables + integer_variables + continuous_variables)

    # Codifica one-hot per le variabili categoriche
    encoder = OneHotEncoder(drop='first')
    encoded_cols = encoder.fit_transform(df[categorical_variables])
    encoded_df = pd.DataFrame(encoded_cols.toarray(), columns=encoder.get_feature_names_out(categorical_variables))

    # Creazione del DataFrame finale
    df_final = pd.concat([df[binary_variables + integer_variables + geodemo], encoded_df], axis=1).fillna(0)

    # Controllo dei valori mancanti
    if df_final.isna().any().any():
        print("Missing values exist.")
    else:
        print("No missing values.")
    
    return df_final

def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola i punteggi di conoscenza, comportamento e atteggiamento e li aggiunge al DataFrame.

    La funzione calcola i punteggi di conoscenza, comportamento, atteggiamento, variabili temporanee per la produzione
    e il punteggio di credito, e crea un punteggio totale per ogni riga. Questi punteggi vengono aggiunti al DataFrame 
    come nuove colonne.

    Parametri:
    - df (pandas.DataFrame): Il DataFrame con le variabili trasformate.

    Ritorna:
    - pandas.DataFrame: Un DataFrame con i punteggi calcolati e una colonna "total_score" che rappresenta la somma di tutti i punteggi.
    """
    # Punteggi di conoscenza
    df["qk3_score"] = (df["qk3_3"] == 1).astype(int)
    df["qk4_score"] = (df["qk4_0"] == 1).astype(int)
    df["qk5_score"] = (df["qk5_102"] == 1).astype(int)
    df["qk6_score"] = (df["qk6_1"] == 1).astype(int)
    df["qk7_1_score"] = (df["qk7_1_1"] == 1).astype(int)
    df["qk7_2_score"] = (df["qk7_2_1"] == 1).astype(int)
    df["qk7_3_score"] = (df["qk7_3_1"] == 1).astype(int)

    # Punteggi comportamentali
    df["qf1_qf2_score"] = (((df["qf1_1"] == 1) | (df["qf1_2"] == 1)) & (df["qf2_1"] == 1)).astype(int)
    df["qf3_score"] = df[["qf3_1", "qf3_4", "qf3_6", "qf3_7", "qf3_8"]].sum(axis=1).gt(0).astype(int)
    df["qf10_1_score"] = df["qf10_1"].isin([1, 2]).astype(int)
    df["qf10_4_score"] = df["qf10_4"].isin([1, 2]).astype(int)
    df["qf10_6_score"] = df["qf10_6"].isin([1, 2]).astype(int)
    df["qf10_7_score"] = df["qf10_7"].isin([1, 2]).astype(int)

    # Variabili temporanee per la produzione
    df["temp_qprod2"] = df[["qprod2_1.0", "qprod2_4.0"]].sum(axis=1).gt(0).astype(int)
    df["temp_qprod3"] = (
        (df[["qprod3_5", "qprod3_6", "qprod3_7", "qprod3_8"]].sum(axis=1) > 0).astype(int) * 2 +
        (df[["qprod3_2", "qprod3_3", "qprod3_4", "qprod3_9", "qprod3_10", "qprod3_11", "qprod3_12", "qprod3_13", "qprod3_18"]].sum(axis=1) > 0).astype(int)
    )

    df["qprod_2pt_score"] = (df["temp_qprod3"] == 2).astype(int)
    df["qprod_1pt_score"] = ((df["temp_qprod2"] == 1) | (df["temp_qprod3"] == 1)).astype(int)

    # Punteggio di credito
    credit_columns = ["qf12_3_e", "qf12_3_f", "qf12_3_g", "qf12_4_k", "qf12_4_l", "qf12_5_m",
                      "qf12_5_o", "qf12_6_p", "qf12_6_q"]
    df["qf12_score"] = (df[credit_columns].sum(axis=1) == 0).astype(int)

    # Punteggio di atteggiamento
    df["qf10_2_score"] = df["qf10_2"].isin([4, 5]).astype(int)
    df["qf10_3_score"] = df["qf10_3"].isin([4, 5]).astype(int)
    df["qf10_8_score"] = df["qf10_8"].isin([4, 5]).astype(int)

    # Definizione delle variabili per FCA
    knowledge_score_variables = ["qk3_score", "qk4_score", "qk5_score", "qk6_score", "qk7_1_score", "qk7_2_score", "qk7_3_score"]
    behavioral_score_variables = ["qf1_qf2_score", "qf3_score", "qf10_1_score", "qf10_4_score", "qf10_6_score", "qf10_7_score", "qprod_2pt_score", "qprod_1pt_score", "qf12_score"]
    attitude_score_variables = ["qf10_2_score", "qf10_3_score", "qf10_8_score"]
    variabili_fca = knowledge_score_variables + behavioral_score_variables + attitude_score_variables

    # Variabili geodemografiche
    geodemo = ["qd1", "AREA5", "qd5b", "generation", "qd10", "education", "qd12", "segmentation"]

    # Creazione del DataFrame finale per l'analisi
    db_analysis = df[geodemo + variabili_fca].copy()
    db_analysis["total_score"] = df[variabili_fca].sum(axis=1)

    return db_analysis


def categorize_generation(age: int) -> str:
    """
    Categorizza una persona in base alla sua fascia di età.

    Parametri:
    - age (int): L'età della persona.

    Ritorna:
    - str: La generazione corrispondente all'età fornita.
    """
    if age >= 80:
        return "Silent_Generation"
    elif 64 <= age <= 79:
        return "Boomers"
    elif 50 <= age <= 63:
        return "Gen_X"
    elif 34 <= age <= 49:
        return "Millennials"
    elif 18 <= age <= 33:
        return "Gen_Z"
    else:
        return "Unknown"


def categorize_education(edu: int) -> str:
    """
    Categorizza una persona in base al suo livello di istruzione.

    Parametri:
    - edu (int): Il livello di istruzione (1, 2, 3, etc.).

    Ritorna:
    - str: La categoria di istruzione ("University", "Diploma", "No_diploma").
    """
    if edu == 1:
        return "University"
    elif edu == 3:
        return "Diploma"
    else:
        return "No_diploma"
    

knowledge_score_variables = ["qk3_score", "qk4_score", "qk5_score", "qk6_score", "qk7_1_score", "qk7_2_score", "qk7_3_score"]
behavioral_score_variables = ["qf1_qf2_score", "qf3_score", "qf10_1_score", "qf10_4_score", "qf10_6_score", "qf10_7_score", "qprod_2pt_score", "qprod_1pt_score", "qf12_score"]
attitude_score_variables = ["qf10_2_score", "qf10_3_score", "qf10_8_score"]
