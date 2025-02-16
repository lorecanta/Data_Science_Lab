import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def check_dataframe_structure(df1, df2):
    """Verifica se due DataFrame hanno la stessa struttura (colonne identiche)."""
    if (df1.columns == df2.columns).all():
        print("PASS")
        return True
    else:
        print("FAIL")
        print("df2: YES , df1: NO", set(df2.columns) - set(df1.columns))
        print("df1: YES , df2: NO", set(df1.columns) - set(df2.columns))
        return False

def transform_dataframe(df):
    """Applica la trasformazione ai dati del DataFrame."""
    df.fillna({"qprod1_d": 99, "qprod2": -99}, inplace=True)
    
    df['qk4'] = np.where(df['qk4'].isin([-97, -99, 0]), df['qk4'], 1)
    df['qk5'] = np.where(df['qk5'].isin([-97, -99, 102]), df['qk5'], 1)
    
    qf10_cols = [col for col in df.columns if col.startswith("qf10_")]
    for col in qf10_cols:
        mode_value = df[col].mode()[0]
        df[col] = df[col].replace([-97, -99], mode_value)
    
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
    
    geodemo = ["AREA5", "qd10", "qd9"]
    categorical_variables = df.columns.difference(binary_variables + integer_variables + continuous_variables)
    
    encoder = OneHotEncoder(drop='first')
    encoded_cols = encoder.fit_transform(df[categorical_variables])
    encoded_df = pd.DataFrame(encoded_cols.toarray(), columns=encoder.get_feature_names_out(categorical_variables))
    
    df_final = pd.concat([df[binary_variables + integer_variables + geodemo], encoded_df], axis=1).fillna(0)
    
    if df_final.isna().any().any():
        print("Missing values exist.")
    else:
        print("No missing values.")
    
    return df_final

knowledge_answer = ["qk3_2","qk3_3","qk3_-97","qk3_1", #qk3
                       "qk4_1","qk4_-97","qk4_0", #qk4
                       "qk5_1","qk5_-97","qk5_102", #qk5
                       "qk6_2","qk6_3","qk6_4","qk6_-97","qk6_1", #qk6
                       "qk7_1_0","qk7_1_-97","qk7_1_1", #qk7_1
                       "qk7_2_0","qk7_2_-97","qk7_2_1", #qk7_2
                       "qk7_3_0","qk7_3_-97","qk7_3_1",] #qk7_3

behevioral_answer =  ["qf1_2","qf1_3","qf1_1", #qf1
                        "qf2_0","qf2_1", #qf2
                        "qf3_1","qf3_3","qf3_4","qf3_6","qf3_7","qf3_8","qf3_99", #qf3
                        "qf10_1", #qf10_1
                        "qf10_4", #qf10_4
                        "qf10_6", #qf10_6
                        "qf10_7", #qf10_7
                        "qprod2_2.0","qprod2_3.0","qprod2_4.0","qprod2_1.0", #qprod2
                        "qprod3_1","qprod3_2","qprod3_3","qprod3_4","qprod3_5","qprod3_6","qprod3_7","qprod3_8","qprod3_9","qprod3_10","qprod3_11","qprod3_12","qprod3_13","qprod3_14","qprod3_15","qprod3_16","qprod3_17","qprod3_18","qprod3_99", #qprod3
                        "qf12_1_a","qf12_1_b","qf12_1_c","qf12_2_d","qf12_3_e","qf12_3_f","qf12_3_g","qf12_4_k","qf12_4_l","qf12_5_m","qf12_5_o","qf12_6_p","qf12_6_q","qf12_7_r","qf12_97","qf12_99" #qf12
                        ]

attitude_answer = ["qf10_2","qf10_3","qf10_8"] #qf10

knowledge_score_variables = [
    "qk3_score", "qk4_score", "qk5_score", "qk6_score", 
    "qk7_1_score", "qk7_2_score", "qk7_3_score"
]

# Behavioral Score
behavioral_score_variables = [
    "qf1_qf2_score", "qf3_score", "qf10_1_score", 
    "qf10_4_score", "qf10_6_score", "qf10_7_score",
    "qprod_2pt_score", "qprod_1pt_score", "qf12_score"
]

# Attitude Score
attitude_score_variables = [
    "qf10_2_score", "qf10_3_score", "qf10_8_score"
]


# Definizione delle generazioni basate sull'anno di nascita
def categorize_generation(age):
    if age >= 80:  # Nati prima del 1945
        return "Silent_Generation"
    elif 64 <= age <= 79:  # Nati tra 1945 e 1960
        return "Boomers"
    elif 50 <= age <= 63:  # Nati tra 1961 e 1974
        return "Gen_X"
    elif 34 <= age <= 49:  # Nati tra 1975 e 1990
        return "Millennials"
    elif 18 <= age <= 33:  # Nati tra 1991 e 2006
        return "Gen_Z"
    else:
        return "Unknown"

# Definizione delle generazioni basate sull'anno di nascita
def categorize_education(edu):
    if edu == 1:  # Nati prima del 1945
        return "University"
    elif edu == 3:  # Nati tra 1945 e 1960
        return "Diploma"
    else:
        return "No_diploma"
    

def calculate_scores(db_final):
    """
    Calcola i punteggi di conoscenza, comportamento e atteggiamento e li aggiunge al DataFrame.
    """
    # Punteggi di conoscenza
    db_final["qk3_score"] = (db_final["qk3_3"] == 1).astype(int)
    db_final["qk4_score"] = (db_final["qk4_0"] == 1).astype(int)
    db_final["qk5_score"] = (db_final["qk5_102"] == 1).astype(int)
    db_final["qk6_score"] = (db_final["qk6_1"] == 1).astype(int)
    db_final["qk7_1_score"] = (db_final["qk7_1_1"] == 1).astype(int)
    db_final["qk7_2_score"] = (db_final["qk7_2_1"] == 1).astype(int)
    db_final["qk7_3_score"] = (db_final["qk7_3_1"] == 1).astype(int)

    # Punteggi comportamentali
    db_final["qf1_qf2_score"] = (((db_final["qf1_1"] == 1) | (db_final["qf1_2"] == 1)) & (db_final["qf2_1"] == 1)).astype(int)
    db_final["qf3_score"] = db_final[["qf3_1", "qf3_4", "qf3_6", "qf3_7", "qf3_8"]].sum(axis=1).gt(0).astype(int)
    db_final["qf10_1_score"] = db_final["qf10_1"].isin([1, 2]).astype(int)
    db_final["qf10_4_score"] = db_final["qf10_4"].isin([1, 2]).astype(int)
    db_final["qf10_6_score"] = db_final["qf10_6"].isin([1, 2]).astype(int)
    db_final["qf10_7_score"] = db_final["qf10_7"].isin([1, 2]).astype(int)

    # Variabili temporanee per la produzione
    db_final["temp_qprod2"] = db_final[["qprod2_1.0", "qprod2_4.0"]].sum(axis=1).gt(0).astype(int)
    db_final["temp_qprod3"] = (
        (db_final[["qprod3_5", "qprod3_6", "qprod3_7", "qprod3_8"]].sum(axis=1) > 0).astype(int) * 2 +
        (db_final[["qprod3_2", "qprod3_3", "qprod3_4", "qprod3_9", "qprod3_10", "qprod3_11", "qprod3_12", "qprod3_13", "qprod3_18"]].sum(axis=1) > 0).astype(int)
    )

    db_final["qprod_2pt_score"] = (db_final["temp_qprod3"] == 2).astype(int)
    db_final["qprod_1pt_score"] = ((db_final["temp_qprod2"] == 1) | (db_final["temp_qprod3"] == 1)).astype(int)

    # Punteggio di credito
    credit_columns = ["qf12_3_e", "qf12_3_f", "qf12_3_g", "qf12_4_k", "qf12_4_l", "qf12_5_m",
                      "qf12_5_o", "qf12_6_p", "qf12_6_q"]
    db_final["qf12_score"] = (db_final[credit_columns].sum(axis=1) == 0).astype(int)

    # Punteggio di atteggiamento
    db_final["qf10_2_score"] = db_final["qf10_2"].isin([4, 5]).astype(int)
    db_final["qf10_3_score"] = db_final["qf10_3"].isin([4, 5]).astype(int)
    db_final["qf10_8_score"] = db_final["qf10_8"].isin([4, 5]).astype(int)

    # Definizione delle variabili per FCA
    knowledge_score_variables = ["qk3_score", "qk4_score", "qk5_score", "qk6_score", "qk7_1_score", "qk7_2_score", "qk7_3_score"]
    behavioral_score_variables = ["qf1_qf2_score", "qf3_score", "qf10_1_score", "qf10_4_score", "qf10_6_score", "qf10_7_score", "qprod_2pt_score", "qprod_1pt_score", "qf12_score"]
    attitude_score_variables = ["qf10_2_score", "qf10_3_score", "qf10_8_score"]
    variabili_fca = knowledge_score_variables + behavioral_score_variables + attitude_score_variables

    # Variabili geodemografiche
    geodemo = ["qd1", "AREA5", "qd5b", "generation", "qd10", "education", "qd12", "segmentation"]

    # Creazione del DataFrame finale per l'analisi
    db_analysis = db_final[geodemo + variabili_fca].copy()
    db_analysis["total_score"] = db_final[variabili_fca].sum(axis=1)

    return db_analysis


def create_train_test_set(data: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
    """
    Funzione per creare i set di train e test da un DataFrame.
    
    :param data: DataFrame contenente i dati del database
    :param test_size: Percentuale di dati da riservare per il test (default 0.2 = 20%)
    :param random_state: Semenza per la randomizzazione della divisione dei dati (default 42)
    
    :return: due DataFrame (train_set, test_set)
    """
    
    # Creare il train e test set utilizzando train_test_split
    train_set, test_set = train_test_split(data, test_size=test_size, random_state=random_state)
    
    return train_set, test_set
