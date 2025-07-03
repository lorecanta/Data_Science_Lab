import pandas as pd
import numpy as np
import itertools
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging
from helpers.models import AssociationRules

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
    min_support: float = 0.1,
    min_confidence: float = 0.6,
    min_lift: float = 1.7,
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
