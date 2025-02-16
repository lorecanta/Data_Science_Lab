import pandas as pd
import numpy as np
import itertools
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt

class AssociationRules:
    def __init__(self, data: pd.DataFrame, columns_A: list, columns_B: list, group: str):
        """
        data: DataFrame principale
        columns_A: colonne per l'insieme A
        columns_B: colonne per l'insieme B
        group: stringa per la colonna 'group' nell'output finale
        """
        self.data = data
        self.columns_A = columns_A
        self.columns_B = columns_B
        self.group = group  # Nuovo parametro 'group'

    def support(self, itemset):
        """
        Calcola il supporto di un insieme di item.
        """
        return (self.data[itemset].all(axis=1).sum()) / len(self.data)

    def confidence(self, itemset_A, itemset_B):
        """
        Calcola la confidenza della regola A -> B.
        """
        support_A_and_B = (self.data[itemset_A + itemset_B].all(axis=1).sum()) / len(self.data)
        support_A = self.support(itemset_A)
        return support_A_and_B / support_A if support_A != 0 else 0

    def lift(self, itemset_A, itemset_B):
        """
        Calcola il lift della regola A -> B.
        """
        support_A_and_B = (self.data[itemset_A + itemset_B].all(axis=1).sum()) / len(self.data)
        support_A = self.support(itemset_A)
        support_B = self.support(itemset_B)

        if support_A != 0 and support_B != 0:
            return support_A_and_B / (support_A * support_B)
        else:
            return 0

    def get_combinations(self, itemset):
        """
        Genera tutte le combinazioni di sottoinsiemi per un dato insieme, senza alcuna dimensione massima.
        """
        all_combinations = []

        # Genera tutte le combinazioni di sottoinsiemi senza limitazione di dimensione
        for size in range(1, len(itemset) + 1):
            for combination in itertools.combinations(itemset, size):
                all_combinations.append(combination)

        return all_combinations

    def filter_duplicate_combinations(self, combinations, itemset):
        """
        Filtra le combinazioni che non si verificano nel dataset.
        """
        valid_combinations = []
        for combination in combinations:
            # Verifica che la combinazione effettivamente esista nel dataset
            if self.data[list(combination)].all(axis=1).any():
                valid_combinations.append(combination)
        return valid_combinations

    def calculate_metrics_for_pair(self, itemset_A, itemset_B):
        """
        Calcola Supporto, Confidenza e Lift per una coppia di insiemi A e B.
        """
        support_A_and_B = self.support(list(itemset_A) + list(itemset_B))
        confidence = self.confidence(list(itemset_A), list(itemset_B))
        lift = self.lift(list(itemset_A), list(itemset_B))

        result = {
            'A': itemset_A,
            'B': itemset_B,
            'Support': support_A_and_B,
            'Confidence': confidence,
            'Lift': lift,
            'Group': self.group  # Aggiungi il group a ciascun risultato
        }

        return result

    def calculate_all_metrics_for_selected_sets(self):
        """
        Calcola Supporto, Confidenza e Lift per le combinazioni selezionate di A e B.
        """
        # Seleziona le colonne valide per A e B
        valid_columns_A = self.data.columns.intersection(self.columns_A).tolist()
        valid_columns_B = self.data.columns.intersection(self.columns_B).tolist()

        # Verifica che le colonne non siano vuote
        if len(valid_columns_A) == 0 or len(valid_columns_B) == 0:
            print("Nessuna combinazione possibile: colonne vuote.")
            return pd.DataFrame()  # Nessuna combinazione possibile

        # Genera combinazioni per A e B senza limite sulla dimensione
        combinations_A = self.get_combinations(valid_columns_A)
        combinations_B = self.get_combinations(valid_columns_B)

        # Filtra le combinazioni duplicate (quelle che non si verificano nel dataset)
        combinations_A = self.filter_duplicate_combinations(combinations_A, valid_columns_A)
        combinations_B = self.filter_duplicate_combinations(combinations_B, valid_columns_B)

        # Calcola le metriche in parallelo
        results = Parallel(n_jobs=-1)(  # Parallelo per calcolare le metriche in parallelo
            delayed(self.calculate_metrics_for_pair)(itemset_A, itemset_B)
            for itemset_A in combinations_A
            for itemset_B in combinations_B
            if set(itemset_A).isdisjoint(itemset_B)  # A e B devono essere disgiunti
        )

        return pd.DataFrame(results)

    def filter_by_values(self, df, min_support=0, min_confidence=0, min_lift=0):
        """
        Filtra le regole di associazione in base ai valori fissi per Support, Confidence e Lift.
        Ogni metrica ha un valore minimo separato.
        """
        # Filtra i risultati in base ai valori minimi
        filtered_df = df[
            (df['Support'] >= min_support) & 
            (df['Confidence'] >= min_confidence) & 
            (df['Lift'] >= min_lift)
        ]

        return filtered_df
    


def plot_metrics_distribution(df):
    """
    Funzione per creare il grafico della distribuzione delle metriche di Lift, Confidence e Support.
    df: DataFrame contenente le colonne 'Support', 'Confidence' e 'Lift'.
    """
    # Impostare il tema di seaborn
    sns.set(style="whitegrid")
    
    # Creazione della figura
    plt.figure(figsize=(12, 8))
    
    # Distribuzione del Support
    plt.subplot(3, 1, 1)  # 3 righe, 1 colonna, 1° grafico
    sns.histplot(df['Support'], kde=True, color='blue', bins=50)
    plt.title('Distribuzione del Support')
    plt.xlabel('Support')
    plt.ylabel('Densità')

    # Distribuzione della Confidence
    plt.subplot(3, 1, 2)  # 3 righe, 1 colonna, 2° grafico
    sns.histplot(df['Confidence'], kde=True, color='green', bins=50)
    plt.title('Distribuzione della Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Densità')

    # Distribuzione del Lift
    plt.subplot(3, 1, 3)  # 3 righe, 1 colonna, 3° grafico
    sns.histplot(df['Lift'], kde=True, color='red', bins=50)
    plt.title('Distribuzione del Lift')
    plt.xlabel('Lift')
    plt.ylabel('Densità')

    # Mostra il grafico
    plt.tight_layout()  # Per evitare sovrapposizioni dei grafici
    plt.show()


import pandas as pd

def analyze_association_rules(
    df_train, df_test, segmentation_column, segmentation_values,
    columns_A, columns_B, export_name, min_support=0.12, min_confidence=0.6, min_lift=1.35
):
    all_results = []
    
    # Itera su tutti i segmenti
    for segment in segmentation_values:
        df_iter = df_train[df_train[segmentation_column] == segment]
        
        # Crea l'oggetto AssociationRules
        ar = AssociationRules(df_iter, columns_A=columns_A, columns_B=columns_B, group=segment)
        
        # Calcola le metriche
        results = ar.calculate_all_metrics_for_selected_sets()
        
        # Filtra le regole
        filtered_results = ar.filter_by_values(results, min_support, min_confidence, min_lift)
        
        all_results.append(filtered_results)
    
    # Concatena tutti i risultati
    final_results_db = pd.concat(all_results, ignore_index=True)
    
    # Prepara i dati per il test
    lista_A = final_results_db["A"].tolist()
    lista_B = final_results_db["B"].tolist()
    
    support_test_values, confidence_test_values, lift_test_values = [], [], []
    
    ar_test = AssociationRules(df_test, columns_A=columns_A, columns_B=columns_B, group="test")
    
    for item_a, item_b in zip(lista_A, lista_B):
        support_test_values.append(ar_test.support(list(item_a + item_b)))
        confidence_test_values.append(ar_test.confidence(list(item_a), list(item_b)))
        lift_test_values.append(ar_test.lift(list(item_a), list(item_b)))
    
    final_results_db["supporto_test"] = support_test_values
    final_results_db["confidenza_test"] = confidence_test_values
    final_results_db["lift_test"] = lift_test_values
    
    # Creazione del riassunto per "Group"
    df_summary = final_results_db.groupby("Group").agg({
        "Support": "mean",
        "Confidence": "mean",
        "Lift": "mean",
        "supporto_test": "mean",
        "confidenza_test": "mean",
        "lift_test": "mean"
    }).reset_index()
    
    # Rinomina le colonne
    df_summary.rename(columns={
        "Support": "Support Medio",
        "Confidence": "Confidence Media",
        "Lift": "Lift Medio",
        "supporto_test": "Support Test Medio",
        "confidenza_test": "Confidence Test Media",
        "lift_test": "Lift Test Medio"
    }, inplace=True)
    
    # Esporta i risultati
    final_results_db.to_csv(f"{export_name}.csv", index=False)
    return final_results_db, df_summary
