import pandas as pd
from helpers import utils_data_preparation
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_financial_literacy_data(file_path: str) -> pd.DataFrame:
    """
    Loads, transforms, and calculates scores on the financial literacy dataset.

    This function performs the following steps:
    1. Loads data from a CSV file.
    2. Cleans and prepares the data using individual transformation functions.
    3. Categorizes education and generation for each record.
    4. Creates a "segmentation" variable by combining education and generation categories.
    5. Calculates scores for specific variables (knowledge, behavior, attitude, etc.).

    Parameters:
    - file_path (str): The path to the CSV file containing financial literacy data.

    Returns:
    - pandas.DataFrame: A transformed DataFrame with additional columns for "education", "generation",
      "segmentation", and calculated scores.
    """
    logging.info("Starting financial literacy data processing...")

    # 1. Load data from the CSV file
    logging.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    logging.info("Data loaded successfully.")
    

    # 2. Apply individual transformation steps
    logging.info("Applying data transformation steps...")
    df = utils_data_preparation.handle_missing_values(df)
    df = utils_data_preparation.modify_columns(df)

    # Replace -97 and -99 in 'qf10_' columns with the mode
    qf10_cols = [col for col in df.columns if col.startswith("qf10_")]
    df = utils_data_preparation.replace_with_mode(df, qf10_cols)

    # Define variable groups
    integer_variables = ["qd5b", "qd7"] + qf10_cols
    categorical_variables = df.columns.difference(
        utils_data_preparation.binary_variables + integer_variables + utils_data_preparation.continuous_variables
    )

    # Apply one-hot encoding to categorical variables
    encoded_df = utils_data_preparation.encode_categorical_variables(df, categorical_variables)

    # Create the final DataFrame
    df = pd.concat(
        [df[utils_data_preparation.binary_variables + integer_variables + utils_data_preparation.geodemo_orig], encoded_df],
        axis=1
    ).fillna(0)

    # 3. Categorize education and generation
    logging.info("Categorizing education and generation...")
    df["education"] = df["qd9"].apply(utils_data_preparation.categorize_education)
    df["generation"] = df["qd7"].apply(utils_data_preparation.categorize_generation)

    # Check for missing values before applying fillna(0)
    if df.isna().any().any():
        logging.warning("Missing values exist in the DataFrame. Displaying missing values:")
        logging.warning(df[df.isna().any(axis=1)])
    else:
        logging.info("No missing values found in the DataFrame.")

    logging.info("Data transformation completed.")

    return df

def analysis_dataframe_preparation(df: pd.DataFrame, var1: str, var2: str, min_weight: int, n_folds: int, var_output: str = "segmentation") -> pd.DataFrame:
    """
    Prepares the analysis DataFrame by creating a segmentation variable, calculating scores, 
    and adding a random variable for cross-validation.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.
    - var1 (str): The first variable to combine for segmentation.
    - var2 (str): The second variable to combine for segmentation.
    - min_weight (int): The minimum weight for filtering segmentation values.
    - n_folds (int): The number of folds for cross-validation.
    - var_output (str): The name of the output segmentation variable. Default is "segmentation".

    Returns:
    - pandas.DataFrame: The prepared DataFrame with segmentation, calculated scores, and a random variable for cross-validation.
    """
    # 4. Create the 'segmentation' variable as a combination of 'var1' and 'var2'
    logging.info(f"Creating '{var_output}' variable...")
    df = utils_data_preparation.create_combined_variable(df, var1, var2, var_output)
    logging.info(f"Variable '{var_output}' created successfully.")

    # 5. Calculate scores for specific variables (knowledge, behavior, attitude, etc.)
    logging.info("Calculating scores...")
    df = utils_data_preparation.calculate_all_scores(df)
    logging.info("Score calculation completed.")

    # Create the analysis DataFrame
    logging.info("Creating analysis DataFrame...")
    df_analysis = utils_data_preparation.create_db_analysis(df)
    logging.info("Analysis DataFrame created successfully.")

    # Log segmentation details
    segmentation_counts = df_analysis[var_output].value_counts()
    logging.info(f"Segmentation counts: {segmentation_counts.to_dict()}")

    # Filter segmentation values based on minimum weight
    segmentation_values = segmentation_counts[segmentation_counts >= min_weight].index.tolist()
    logging.info(f"Segmentation values with minimum weight ({min_weight}): {segmentation_values}")

    # Add a random variable for cross-validation
    logging.info(f"Adding a random variable for {n_folds}-fold cross-validation...")
    df_analysis["cv_fold"] = np.random.randint(0, n_folds, size=len(df_analysis))
    logging.info(f"Random variable for cross-validation added successfully.")

    logging.info("Analysis DataFrame preparation completed.")
    return df, df_analysis, segmentation_values
