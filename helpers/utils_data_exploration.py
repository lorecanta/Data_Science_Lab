import pandas as pd

def convert_column_types(df, type_map):
    """
    Converte i tipi di colonna di un DataFrame in base a un dizionario di tipi.

    Parametri:
    - df (pd.DataFrame): il DataFrame da convertire
    - type_map (dict): dizionario con {colonna: tipo}, dove tipo è una stringa tra:
        'int', 'float', 'category', 'bool', 'str'

    Ritorna:
    - pd.DataFrame con tipi convertiti
    """
    for col, col_type in type_map.items():
        if col in df.columns:
            try:
                if col_type == 'category':
                    df[col] = df[col].astype('category')
                elif col_type == 'bool':
                    df[col] = df[col].astype('boolean')  # usa pandas nullable boolean
                elif col_type == 'int':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')  # nullable integer
                elif col_type == 'float':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif col_type == 'str':
                    df[col] = df[col].astype(str)
            except Exception as e:
                print(f"Errore nella conversione della colonna '{col}' a '{col_type}': {e}")
    return df

def generate_variable_summary(df, type_map):
    summary = []

    for col, col_type in type_map.items():
        if col in df.columns:
            na_pct = df[col].isna().mean() * 100
            kpi = None

            if col_type == 'int' or col_type == 'float':
                col_data = df[col]
                kpi = {
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'median': col_data.median()
                }
            elif col_type == 'category' or col_type == 'str':
                col_data = df[col].astype('category')
                mode = col_data.mode()
                kpi = {
                    'mode': mode.iloc[0] if not mode.empty else None,
                    'n_unique': col_data.nunique()
                }
            elif col_type == 'bool':
                col_data = df[col]
                true_pct = (col_data == True).mean() * 100
                false_pct = (col_data == False).mean() * 100
                kpi = {
                    '% True': true_pct,
                    '% False': false_pct
                }

            summary.append({
                'variable': col,
                'type': col_type,
                '% NA': round(na_pct, 2),
                'kpi': kpi
            })

    return pd.DataFrame(summary)

# Esegui la funzione sul DataFrame (df) e mapping (column_types)
# df deve già essere stato caricato e convertito
try:
    variable_summary = generate_variable_summary(df, type_map)
except NameError:
    variable_summary = "⚠️ Il DataFrame `df` non è definito in questo ambiente."

variable_summary if isinstance(variable_summary, pd.DataFrame) else variable_summary