import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Static variable definitions
binary_variables = [
    "qd1", "SM", "qd12", "qprod1c_1", "qprod1c_2", "qprod1c_3", "qprod1c_5", "qprod1c_6", "qprod1c_7",
    "qprod1c_8", "qprod1c_10", "qprod1c_11", "qprod1c_12", "qprod1c_14", "qprod1c_99", "qf3_1", "qf3_3", "qf3_4",
    "qf3_6", "qf3_7", "qf3_8", "qf3_99", "qf9_1", "qf9_10", "qf9_2", "qf9_3", "qf9_4", "qf9_5", "qf9_6", "qf9_7",
    "qf9_8", "qf9_9", "qf9_99", "qprod3_1", "qprod3_2", "qprod3_3", "qprod3_4", "qprod3_5", "qprod3_6", "qprod3_7",
    "qprod3_8", "qprod3_9", "qprod3_10", "qprod3_11", "qprod3_12", "qprod3_13", "qprod3_14", "qprod3_15", "qprod3_16",
    "qprod3_17", "qprod3_18", "qprod3_99", "qf12_1_a", "qf12_1_b", "qf12_1_c", "qf12_2_d", "qf12_3_e", "qf12_3_f",
    "qf12_3_g", "qf12_4_k", "qf12_4_l", "qf12_5_m", "qf12_5_o", "qf12_6_p", "qf12_6_q", "qf12_7_r", "qf12_97", "qf12_99"
]
continuous_variables = ["pesofitc"]
geodemo_orig = ["AREA5", "qd10", "qd9"]

# Define score variables
knowledge_score_variables = ["qk3_score", "qk4_score", "qk5_score", "qk6_score", "qk7_1_score", "qk7_2_score", "qk7_3_score"]
behavioral_score_variables = ["qf1_qf2_score", "qf3_score", "qf10_1_score", "qf10_4_score", "qf10_6_score", "qf10_7_score", "qprod_2pt_score", "qprod_1pt_score", "qf12_score"]
attitude_score_variables = ["qf10_2_score", "qf10_3_score", "qf10_8_score"]
variabili_fca = knowledge_score_variables + behavioral_score_variables + attitude_score_variables

# Geodemographic variables
geodemo = ["qd1", "AREA5", "qd5b", "generation", "qd10", "education", "qd12", "segmentation"]

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the DataFrame by filling specific columns with predefined values.

    This function replaces missing values in the columns `qprod1_d` and `qprod2` with the values `99` and `-99`, respectively.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.

    Returns:
    - pandas.DataFrame: The DataFrame with missing values handled.
    """
    logging.info("Handling missing values...")
    df.fillna({"qprod1_d": 99, "qprod2": -99}, inplace=True)
    return df

def modify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modifies specific columns based on predefined conditions.

    This function updates the values in the columns `qk4` and `qk5`:
    - For `qk4`, values not in `[-97, -99, 0]` are replaced with `1`.
    - For `qk5`, values not in `[-97, -99, 102]` are replaced with `1`.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.

    Returns:
    - pandas.DataFrame: The DataFrame with modified columns.
    """
    logging.info("Modifying specific columns...")
    df['qk4'] = np.where(df['qk4'].isin([-97, -99, 0]), df['qk4'], 1)
    df['qk5'] = np.where(df['qk5'].isin([-97, -99, 102]), df['qk5'], 1)
    return df

def replace_with_mode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Replaces specific values (-97, -99) in the given columns with the mode of each column.

    This function iterates over the provided list of columns and replaces occurrences of `-97` and `-99` with the mode (most frequent value) of the respective column.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.
    - columns (list): List of column names to process.

    Returns:
    - pandas.DataFrame: The DataFrame with replaced values.
    """
    logging.info(f"Replacing -97 and -99 with mode for columns: {columns}")
    for col in columns:
        mode_value = df[col].mode()[0]
        df[col] = df[col].replace([-97, -99], mode_value)
    return df

def encode_categorical_variables(df: pd.DataFrame, categorical_variables: list) -> pd.DataFrame:
    """
    Applies one-hot encoding to the categorical variables.

    This function uses `OneHotEncoder` from `sklearn` to encode the specified categorical variables. The first category is dropped to avoid multicollinearity.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.
    - categorical_variables (list): List of categorical variable names.

    Returns:
    - pandas.DataFrame: A DataFrame with one-hot encoded variables.
    """
    logging.info(f"Applying one-hot encoding to categorical variables...")
    encoder = OneHotEncoder(drop='first')
    encoded_cols = encoder.fit_transform(df[categorical_variables])
    encoded_df = pd.DataFrame(encoded_cols.toarray(), columns=encoder.get_feature_names_out(categorical_variables))
    return encoded_df

def calculate_knowledge_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates knowledge-related scores and adds them to the DataFrame.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.

    Returns:
    - pandas.DataFrame: The DataFrame with knowledge scores added.
    """
    df["qk3_score"] = (df["qk3_3"] == 1).astype(int)
    df["qk4_score"] = (df["qk4_0"] == 1).astype(int)
    df["qk5_score"] = (df["qk5_102"] == 1).astype(int)
    df["qk6_score"] = (df["qk6_1"] == 1).astype(int)
    df["qk7_1_score"] = (df["qk7_1_1"] == 1).astype(int)
    df["qk7_2_score"] = (df["qk7_2_1"] == 1).astype(int)
    df["qk7_3_score"] = (df["qk7_3_1"] == 1).astype(int)
    return df

def calculate_behavioral_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates behavioral-related scores and adds them to the DataFrame.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.

    Returns:
    - pandas.DataFrame: The DataFrame with behavioral scores added.
    """
    df["qf1_qf2_score"] = (((df["qf1_1"] == 1) | (df["qf1_2"] == 1)) & (df["qf2_1"] == 1)).astype(int)
    df["qf3_score"] = df[["qf3_1", "qf3_4", "qf3_6", "qf3_7", "qf3_8"]].sum(axis=1).gt(0).astype(int)
    df["qf10_1_score"] = df["qf10_1"].isin([1, 2]).astype(int)
    df["qf10_4_score"] = df["qf10_4"].isin([1, 2]).astype(int)
    df["qf10_6_score"] = df["qf10_6"].isin([1, 2]).astype(int)
    df["qf10_7_score"] = df["qf10_7"].isin([1, 2]).astype(int)
    return df

def calculate_production_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates temporary production-related scores and adds them to the DataFrame.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.

    Returns:
    - pandas.DataFrame: The DataFrame with production scores added.
    """
    df["temp_qprod2"] = df[["qprod2_1.0", "qprod2_4.0"]].sum(axis=1).gt(0).astype(int)
    df["temp_qprod3"] = (
        (df[["qprod3_5", "qprod3_6", "qprod3_7", "qprod3_8"]].sum(axis=1) > 0).astype(int) * 2 +
        (df[["qprod3_2", "qprod3_3", "qprod3_4", "qprod3_9", "qprod3_10", "qprod3_11", "qprod3_12", "qprod3_13", "qprod3_18"]].sum(axis=1) > 0).astype(int)
    )
    df["qprod_2pt_score"] = (df["temp_qprod3"] == 2).astype(int)
    df["qprod_1pt_score"] = ((df["temp_qprod2"] == 1) | (df["temp_qprod3"] == 1)).astype(int)
    return df

def calculate_credit_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates credit-related scores and adds them to the DataFrame.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.

    Returns:
    - pandas.DataFrame: The DataFrame with credit scores added.
    """
    credit_columns = ["qf12_3_e", "qf12_3_f", "qf12_3_g", "qf12_4_k", "qf12_4_l", "qf12_5_m",
                      "qf12_5_o", "qf12_6_p", "qf12_6_q"]
    df["qf12_score"] = (df[credit_columns].sum(axis=1) == 0).astype(int)
    return df

def calculate_attitude_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates attitude-related scores and adds them to the DataFrame.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.

    Returns:
    - pandas.DataFrame: The DataFrame with attitude scores added.
    """
    df["qf10_2_score"] = df["qf10_2"].isin([4, 5]).astype(int)
    df["qf10_3_score"] = df["qf10_3"].isin([4, 5]).astype(int)
    df["qf10_8_score"] = df["qf10_8"].isin([4, 5]).astype(int)
    return df

def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates knowledge, behavior, attitude, production, and credit scores and adds them to the DataFrame.

    This function combines the results of individual score calculation functions.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.

    Returns:
    - pandas.DataFrame: A DataFrame with all scores added.
    """
    logging.info("Calculating all scores...")

    # Calculate individual scores
    df = calculate_knowledge_scores(df)
    df = calculate_behavioral_scores(df)
    df = calculate_production_scores(df)
    df = calculate_credit_scores(df)
    df = calculate_attitude_scores(df)

    logging.info("All scores calculated successfully.")
    return df

def create_db_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the `db_analysis` DataFrame for analysis.

    This function selects geodemographic variables and calculated scores, and computes the total score.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame with calculated scores.

    Returns:
    - pandas.DataFrame: A DataFrame containing geodemographic variables, scores, and the total score.
    """
    logging.info("Creating db_analysis DataFrame...")

    # Select geodemographic variables and calculated scores
    db_analysis = df[geodemo + variabili_fca].copy()

    # Calculate the total score
    db_analysis["total_score"] = df[variabili_fca].sum(axis=1)

    logging.info("db_analysis DataFrame created successfully.")
    return db_analysis

def categorize_generation(age: int) -> str:
    """
    Categorizes a person based on their age group.

    This function assigns a generation label based on the provided age:
    - `Silent_Generation`: Age >= 80
    - `Boomers`: 64 <= Age <= 79
    - `Gen_X`: 50 <= Age <= 63
    - `Millennials`: 34 <= Age <= 49
    - `Gen_Z`: 18 <= Age <= 33
    - `Unknown`: Age < 18 or invalid age

    Parameters:
    - age (int): The age of the person.

    Returns:
    - str: The generation corresponding to the given age.
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
    Categorizes a person based on their education level.

    This function assigns an education category based on the provided level:
    - `University`: Level 1
    - `Diploma`: Level 3
    - `No_diploma`: Any other level

    Parameters:
    - edu (int): The education level (1, 2, 3, etc.).

    Returns:
    - str: The education category ("University", "Diploma", "No_diploma").
    """
    if edu == 1:
        return "University"
    elif edu == 3:
        return "Diploma"
    else:
        return "No_diploma"

def create_combined_variable(df: pd.DataFrame, col1: str, col2: str, new_col_name: str) -> pd.DataFrame:
    """
    Creates a new column by combining two existing columns with an underscore.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.
    - col1 (str): The name of the first column to combine.
    - col2 (str): The name of the second column to combine.
    - new_col_name (str): The name of the new column to create.

    Returns:
    - pandas.DataFrame: The DataFrame with the new combined column added.
    """
    logging.info(f"Creating combined variable '{new_col_name}' from '{col1}' and '{col2}'...")
    df[new_col_name] = df[col1].astype(str) + "_" + df[col2].astype(str)
    logging.info(f"Combined variable '{new_col_name}' created successfully.")
    return df

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
    df = handle_missing_values(df)
    df = modify_columns(df)

    # Replace -97 and -99 in 'qf10_' columns with the mode
    qf10_cols = [col for col in df.columns if col.startswith("qf10_")]
    df = replace_with_mode(df, qf10_cols)

    # Define variable groups
    integer_variables = ["qd5b", "qd7"] + qf10_cols
    categorical_variables = df.columns.difference(
        binary_variables + integer_variables + continuous_variables
    )

    # Apply one-hot encoding to categorical variables
    encoded_df = encode_categorical_variables(df, categorical_variables)

    # Create the final DataFrame
    df = pd.concat(
        [df[binary_variables + integer_variables + geodemo_orig], encoded_df],
        axis=1
    ).fillna(0)

    # 3. Categorize education and generation
    logging.info("Categorizing education and generation...")
    df["education"] = df["qd9"].apply(categorize_education)
    df["generation"] = df["qd7"].apply(categorize_generation)

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
    df = create_combined_variable(df, var1, var2, var_output)
    logging.info(f"Variable '{var_output}' created successfully.")

    # 5. Calculate scores for specific variables (knowledge, behavior, attitude, etc.)
    logging.info("Calculating scores...")
    df = calculate_all_scores(df)
    logging.info("Score calculation completed.")

    # Create the analysis DataFrame
    logging.info("Creating analysis DataFrame...")
    df_analysis = create_db_analysis(df)
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

    df_analysis_invertito=df_analysis.copy()
    colonne_da_invertire=attitude_score_variables+knowledge_score_variables+behavioral_score_variables
    df_analysis_invertito[colonne_da_invertire] = 1 - df_analysis_invertito[colonne_da_invertire]

    risultato = {
        "df": df,
        "df_analysis": df_analysis,
        "segmentation_values": segmentation_values,
        "df_analysis_invertito":df_analysis_invertito
    }
    return risultato

