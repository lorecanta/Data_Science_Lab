import pandas as pd
import numpy as np
import itertools
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
import os

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
    


def plot_metrics_distribution(df, save_path="plots"):
    """
    Function to create and save the plot for the distribution of the metrics Lift, Confidence, and Support.
    df: DataFrame containing the columns 'Support', 'Confidence' and 'Lift'.
    save_path: Directory to save the image.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Set a more stylish theme for seaborn
    sns.set_palette("Set2")  # Use a nice palette of colors
    sns.set(style="whitegrid", font_scale=1.2)  # Increase font scale for better readability
    
    # Create the figure
    plt.figure(figsize=(12, 10))  # Increased height for more space between plots
    
    # Function to compute and display statistics
    def show_statistics(ax, data, metric_name):
        mean = np.mean(data)
        median = np.median(data)
        std_dev = np.std(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        min_val = np.min(data)
        max_val = np.max(data)

        # Annotate statistics on the plot
        stats_text = (
            f'Mean: {mean:.2f}\n'
            f'Median: {median:.2f}\n'
            f'Std Dev: {std_dev:.2f}\n'
            f'25th Percentile: {q1:.2f}\n'
            f'75th Percentile: {q3:.2f}\n'
            f'Min: {min_val:.2f}\n'
            f'Max: {max_val:.2f}'
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top', 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    # Distribution of Support
    ax1 = plt.subplot(3, 1, 1)
    sns.histplot(df['Support'], kde=True, color='blue', bins=50, ax=ax1)
    ax1.set_title('Distribution of Support')
    ax1.set_xlabel('Support')
    ax1.set_ylabel('Density')
    show_statistics(ax1, df['Support'], 'Support')
    
    # Distribution of Confidence
    ax2 = plt.subplot(3, 1, 2)
    sns.histplot(df['Confidence'], kde=True, color='green', bins=50, ax=ax2)
    ax2.set_title('Distribution of Confidence')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Density')
    show_statistics(ax2, df['Confidence'], 'Confidence')
    
    # Distribution of Lift
    ax3 = plt.subplot(3, 1, 3)
    sns.histplot(df['Lift'], kde=True, color='red', bins=50, ax=ax3)
    ax3.set_title('Distribution of Lift')
    ax3.set_xlabel('Lift')
    ax3.set_ylabel('Density')
    show_statistics(ax3, df['Lift'], 'Lift')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{save_path}/metrics_distribution_with_stats.jpg", dpi=300)

    # Close the figure to avoid overlap on future plots
    plt.close()


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
    final_results_db.to_csv(f"output\\{export_name}.csv", index=False, sep = ";")
    return final_results_db, df_summary
