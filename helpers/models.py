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
