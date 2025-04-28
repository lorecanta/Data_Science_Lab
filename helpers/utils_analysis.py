import pandas as pd
import numpy as np
import itertools
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging

class AssociationRules:
    """
    A class to compute association rules and their metrics (Support, Confidence, Lift) 
    between two sets of variables in a dataset.

    Attributes:
    - data (pd.DataFrame): The input dataset.
    - columns_A (list): List of columns representing the first set of variables (A).
    - columns_B (list): List of columns representing the second set of variables (B).
    - group (str): A label to identify the group or segment being analyzed.

    Methods:
    - support(itemset): Calculates the support of a given itemset.
    - confidence(itemset_A, itemset_B): Calculates the confidence of the rule A -> B.
    - lift(itemset_A, itemset_B): Calculates the lift of the rule A -> B.
    - get_combinations(itemset): Generates all possible subsets of a given itemset.
    - filter_duplicate_combinations(combinations, itemset): Filters combinations that do not occur in the dataset.
    - calculate_metrics_for_pair(itemset_A, itemset_B): Computes Support, Confidence, and Lift for a pair of itemsets.
    - calculate_all_metrics_for_selected_sets(): Computes metrics for all valid combinations of A and B.
    - filter_by_values(df, min_support, min_confidence, min_lift): Filters rules based on minimum thresholds for metrics.
    """

    def __init__(self, data: pd.DataFrame, columns_A: list, columns_B: list, group: str):
        """
        Initializes the AssociationRules object.

        Parameters:
        - data (pd.DataFrame): The input dataset.
        - columns_A (list): List of columns for the first set of variables (A).
        - columns_B (list): List of columns for the second set of variables (B).
        - group (str): A label to identify the group or segment being analyzed.
        """
        self.data = data
        self.columns_A = columns_A
        self.columns_B = columns_B
        self.group = group

    def support(self, itemset: list) -> float:
        """
        Calculates the support of a given itemset.

        Parameters:
        - itemset (list): List of column names representing the itemset.

        Returns:
        - float: The support value.
        """
        return self.data[itemset].all(axis=1).mean()

    def confidence(self, itemset_A: list, itemset_B: list) -> float:
        """
        Calculates the confidence of the rule A -> B.

        Parameters:
        - itemset_A (list): List of column names representing the antecedent (A).
        - itemset_B (list): List of column names representing the consequent (B).

        Returns:
        - float: The confidence value.
        """
        support_A_and_B = self.data[itemset_A + itemset_B].all(axis=1).mean()
        support_A = self.support(itemset_A)
        return support_A_and_B / support_A if support_A != 0 else 0

    def lift(self, itemset_A: list, itemset_B: list) -> float:
        """
        Calculates the lift of the rule A -> B.

        Parameters:
        - itemset_A (list): List of column names representing the antecedent (A).
        - itemset_B (list): List of column names representing the consequent (B).

        Returns:
        - float: The lift value.
        """
        support_A_and_B = self.data[itemset_A + itemset_B].all(axis=1).mean()
        support_A = self.support(itemset_A)
        support_B = self.support(itemset_B)
        return support_A_and_B / (support_A * support_B) if support_A != 0 and support_B != 0 else 0

    def get_combinations(self, itemset: list) -> list:
        """
        Generates all possible subsets of a given itemset.

        Parameters:
        - itemset (list): List of column names.

        Returns:
        - list: A list of all possible subsets of the itemset.
        """
        return [combination for size in range(1, len(itemset) + 1) for combination in itertools.combinations(itemset, size)]

    def filter_duplicate_combinations(self, combinations: list, itemset: list) -> list:
        """
        Filters combinations that do not occur in the dataset.

        Parameters:
        - combinations (list): List of combinations to filter.
        - itemset (list): List of column names representing the itemset.

        Returns:
        - list: A list of valid combinations.
        """
        return [combination for combination in combinations if self.data[list(combination)].all(axis=1).any()]

    def calculate_metrics_for_pair(self, itemset_A: tuple, itemset_B: tuple) -> dict:
        """
        Computes Support, Confidence, and Lift for a pair of itemsets.

        Parameters:
        - itemset_A (tuple): Tuple representing the antecedent (A).
        - itemset_B (tuple): Tuple representing the consequent (B).

        Returns:
        - dict: A dictionary containing the metrics and the group label.
        """
        return {
            'A': itemset_A,
            'B': itemset_B,
            'Support': self.support(list(itemset_A) + list(itemset_B)),
            'Confidence': self.confidence(list(itemset_A), list(itemset_B)),
            'Lift': self.lift(list(itemset_A), list(itemset_B)),
            'Group': self.group
        }

    def calculate_all_metrics_for_selected_sets(self) -> pd.DataFrame:
        """
        Computes metrics (Support, Confidence, Lift) for all valid combinations of A and B.

        Returns:
        - pd.DataFrame: A DataFrame containing the metrics for all valid rules.
        """
        # Validate columns
        valid_columns_A = self.data.columns.intersection(self.columns_A).tolist()
        valid_columns_B = self.data.columns.intersection(self.columns_B).tolist()

        if not valid_columns_A or not valid_columns_B:
            logging.warning("No valid combinations: columns are empty.")
            return pd.DataFrame()

        # Generate and filter combinations
        combinations_A = self.filter_duplicate_combinations(self.get_combinations(valid_columns_A), valid_columns_A)
        combinations_B = self.filter_duplicate_combinations(self.get_combinations(valid_columns_B), valid_columns_B)

        # Calculate metrics in parallel
        results = Parallel(n_jobs=-1)(
            delayed(self.calculate_metrics_for_pair)(itemset_A, itemset_B)
            for itemset_A in combinations_A
            for itemset_B in combinations_B
            if set(itemset_A).isdisjoint(itemset_B)  # Ensure A and B are disjoint
        )

        return pd.DataFrame(results)

    def filter_by_values(self, df: pd.DataFrame, min_support: float = 0, min_confidence: float = 0, min_lift: float = 0) -> pd.DataFrame:
        """
        Filters rules based on minimum thresholds for Support, Confidence, and Lift.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the rules and their metrics.
        - min_support (float): Minimum support threshold.
        - min_confidence (float): Minimum confidence threshold.
        - min_lift (float): Minimum lift threshold.

        Returns:
        - pd.DataFrame: A filtered DataFrame containing rules that meet the thresholds.
        """
        return df[
            (df['Support'] >= min_support) &
            (df['Confidence'] >= min_confidence) &
            (df['Lift'] >= min_lift)
        ]



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


def _process_segment(segment_data, segment, columns_A, columns_B, min_support, min_confidence, min_lift, n_folds):
    """
    Processes a single segment and performs cross-validation to calculate association rules.

    Parameters:
    - segment_data (pd.DataFrame): The data for the current segment.
    - segment (str): The name of the segment being processed.
    - columns_A (list): The first set of variables (antecedents) for association rules.
    - columns_B (list): The second set of variables (consequents) for association rules.
    - min_support (float): The minimum support threshold for filtering rules.
    - min_confidence (float): The minimum confidence threshold for filtering rules.
    - min_lift (float): The minimum lift threshold for filtering rules.
    - n_folds (int): The number of cross-validation folds.

    Returns:
    - pd.DataFrame: A DataFrame containing the results for all folds in the segment, including metrics for training and testing.
    
    Steps:
    1. Iterates over the specified number of folds.
    2. Splits the data into training and testing sets for each fold.
    3. Calculates association rules on the training set.
    4. Filters the rules based on the specified thresholds.
    5. Validates the filtered rules on the testing set.
    6. Collects the results for each fold.
    """
    logging.info(f"Processing segment: {segment}")
    fold_results = []

    for fold in range(n_folds):
        logging.info(f"Processing fold {fold + 1}/{n_folds} for segment {segment}...")

        # Split data into training and testing sets
        train_data = segment_data[segment_data["cv_fold"] != fold]
        test_data = segment_data[segment_data["cv_fold"] == fold]

        if train_data.empty or test_data.empty:
            logging.warning(f"Fold {fold + 1} for segment {segment} has empty training or testing data.")
            continue

        # Create AssociationRules object for training data
        ar = AssociationRules(train_data, columns_A=columns_A, columns_B=columns_B, group=segment)

        # Calculate metrics for training data
        train_results = ar.calculate_all_metrics_for_selected_sets()

        if train_results.empty:
            logging.warning(f"No rules generated for fold {fold + 1} in segment {segment}.")
            continue

        # Filter rules based on thresholds
        filtered_train_results = ar.filter_by_values(train_results, min_support, min_confidence, min_lift)

        # Validate rules on test data
        ar_test = AssociationRules(test_data, columns_A=columns_A, columns_B=columns_B, group=f"{segment}_fold_{fold}")
        for _, row in filtered_train_results.iterrows():
            support_test = ar_test.support(list(row["A"]) + list(row["B"]))
            confidence_test = ar_test.confidence(list(row["A"]), list(row["B"]))
            lift_test = ar_test.lift(list(row["A"]), list(row["B"]))

            fold_results.append({
                "A": row["A"],
                "B": row["B"],
                "Support": row["Support"],
                "Confidence": row["Confidence"],
                "Lift": row["Lift"],
                "Support_Test": support_test,
                "Confidence_Test": confidence_test,
                "Lift_Test": lift_test,
                "Group": segment
            })

    return pd.DataFrame(fold_results)


def _aggregate_results(all_results):
    """
    Aggregates results across all folds and segments.

    Parameters:
    - all_results (list): A list of DataFrames containing results for each segment and fold.

    Returns:
    - tuple:
        - final_results_db (pd.DataFrame): A DataFrame containing all aggregated results.
        - df_summary (pd.DataFrame): A summary DataFrame with average metrics for each segment.

    Steps:
    1. Concatenates all results into a single DataFrame.
    2. Groups the results by segment and calculates the mean for each metric.
    3. Renames columns in the summary DataFrame for clarity.
    4. Returns the aggregated results and the summary DataFrame.
    """
    if not all_results:
        logging.warning("No results to aggregate.")
        return pd.DataFrame(), pd.DataFrame()

    final_results_db = pd.concat(all_results, ignore_index=True)

    df_summary = final_results_db.groupby("Group").agg({
        "Support": "mean",
        "Confidence": "mean",
        "Lift": "mean",
        "Support_Test": "mean",
        "Confidence_Test": "mean",
        "Lift_Test": "mean"
    }).reset_index()

    df_summary.rename(columns={
        "Support": "Support Mean",
        "Confidence": "Confidence Mean",
        "Lift": "Lift Mean",
        "Support_Test": "Support Test Mean",
        "Confidence_Test": "Confidence Test Mean",
        "Lift_Test": "Lift Test Mean"
    }, inplace=True)

    return final_results_db, df_summary


def analyze_association_rules_cv(
    df: pd.DataFrame,
    segmentation_column: str,
    segmentation_values: list,
    columns_A: list,
    columns_B: list,
    export_name: str,
    min_support: float = 0.12,
    min_confidence: float = 0.6,
    min_lift: float = 1.35,
    n_folds: int = 5
) -> tuple:
    """
    Analyzes association rules between two sets of variables (A and B) across different segments using cross-validation.

    This function performs the following steps:
    1. Iterates over the specified segmentation values.
    2. For each segment, performs cross-validation by iterating over the folds.
    3. For each fold, calculates association rules (Support, Confidence, Lift) between columns_A and columns_B.
    4. Filters the rules based on minimum thresholds for Support, Confidence, and Lift.
    5. Aggregates the results across all folds using the mean values for each metric.
    6. Exports the final aggregated results to a CSV file.

    Parameters:
    - df (pd.DataFrame): The input dataset containing the cross-validation fold assignments.
    - segmentation_column (str): The column used to segment the data (e.g., "segmentation").
    - segmentation_values (list): The list of unique segment values to iterate over.
    - columns_A (list): The first set of variables (antecedents) for association rules.
    - columns_B (list): The second set of variables (consequents) for association rules.
    - export_name (str): The name of the CSV file to export the results.
    - min_support (float): The minimum support threshold for filtering rules. Default is 0.12.
    - min_confidence (float): The minimum confidence threshold for filtering rules. Default is 0.6.
    - min_lift (float): The minimum lift threshold for filtering rules. Default is 1.35.
    - n_folds (int): The number of cross-validation folds. Default is 5.

    Returns:
    - tuple: A tuple containing:
        - final_results_db (pd.DataFrame): The DataFrame with all filtered rules and their aggregated metrics.
        - df_summary (pd.DataFrame): A summary DataFrame with average metrics for each segment.

    Steps:
    1. **Cross-Validation Phase**:
       - For each segment in `segmentation_values`, filter the data for that segment.
       - For each fold, split the data into training (all folds except the current fold) and testing (current fold).
       - Create an `AssociationRules` object and calculate metrics (Support, Confidence, Lift) for all valid combinations of columns_A and columns_B.
       - Filter the rules based on the specified thresholds (min_support, min_confidence, min_lift).
       - Store the metrics for each fold.

    2. **Aggregation**:
       - Aggregate the metrics across all folds by calculating the mean values for each rule.

    3. **Export**:
       - Export the final aggregated results to a CSV file.

    Example Usage:
    ```
    final_results, summary = analyze_association_rules_cv(
        df=data,
        segmentation_column="segmentation",
        segmentation_values=["Segment1", "Segment2"],
        columns_A=["var1", "var2"],
        columns_B=["var3", "var4"],
        export_name="association_rules_cv_results",
        n_folds=5
    )
    ```
    """
    logging.info("Starting cross-validation for association rules analysis.")
    all_results = []

    # Iterate over all segmentation values
    for segment in segmentation_values:
        segment_data = df[df[segmentation_column] == segment]

        if segment_data.empty:
            logging.warning(f"Segment {segment} has no data. Skipping.")
            continue

        fold_results_df = _process_segment(
            segment_data, segment, columns_A, columns_B, min_support, min_confidence, min_lift, n_folds
        )

        if not fold_results_df.empty:
            aggregated_results = fold_results_df.groupby(["A", "B", "Group"]).mean().reset_index()
            all_results.append(aggregated_results)

    final_results_db, df_summary = _aggregate_results(all_results)

    if not final_results_db.empty:
        output_path = f"output/{export_name}.csv"
        final_results_db.to_csv(output_path, index=False, sep=";")
        logging.info(f"Results exported to {output_path}")
    else:
        logging.warning("No results to export.")

    return final_results_db, df_summary
