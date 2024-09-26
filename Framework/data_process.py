# import json
# import pandas as pd
# from typing import Any, Dict, List

# def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
#     items: List[tuple] = []
#     for k, v in d.items():
#         new_key = f"{parent_key}{sep}{k}" if parent_key else k
#         if isinstance(v, dict):
#             items.extend(flatten_dict(v, new_key, sep=sep).items())
#         elif isinstance(v, list):
#             if v and isinstance(v[0], dict):
#                 for i, item in enumerate(v):
#                     items.extend(flatten_dict(item, f"{new_key}{sep}{i}", sep=sep).items())
#             else:
#                 items.append((new_key, json.dumps(v)))
#         else:
#             items.append((new_key, v))
#     return dict(items)

# def load_and_flatten_data(file_path: str) -> pd.DataFrame:
#     with open(file_path, 'r') as f:
#         data = [flatten_dict(json.loads(line)) for line in f]
#     return pd.DataFrame(data)

# def limpiar_dataframe(df):
#     """
#     Limpia un DataFrame eliminando columnas que cumplan ciertas condiciones:
#     - Columnas con más del 20% de valores nulos.
#     - Columnas cuyo nombre contiene la palabra 'id'.
#     - Columnas cuyas celdas contienen únicamente '[]'.

#     Args:
#         df (pd.DataFrame): DataFrame a limpiar.

#     Returns:
#         pd.DataFrame: DataFrame limpio.
#     """

#     # Autoconsistencia: Asegurar que el DataFrame es de tipo Pandas
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("El argumento debe ser un DataFrame de Pandas.")

#     # Eliminar columnas con más del 20% de valores nulos
#     porcentaje_nulos = df.isnull().sum() / len(df)
#     columnas_a_eliminar = porcentaje_nulos[porcentaje_nulos > 0.2].index
#     df.drop(columns=columnas_a_eliminar, inplace=True)

#     # Eliminar columnas con 'id' en el nombre
#     columnas_con_id = df.columns[df.columns.str.contains('id')]
#     df.drop(columns=columnas_con_id, inplace=True)

#     # Eliminar columnas con solo '[]' en todas las celdas
#     columnas_a_eliminar = []
#     for columna in df.columns:
#         validacion = df[columna].apply(lambda x: True if '[]' in str(x) else False ) # Convert to string for checking)
#         if validacion.any():  # Check if any element in 'validacion' is True
#             columnas_a_eliminar.append(columna)
#         validacion = df[columna].apply(lambda x: True if 'http://' in str(x) else False ) # Convert to string for checking)
#         if validacion.any():  # Check if any element in 'validacion' is True
#             columnas_a_eliminar.append(columna)
#         validacion = df[columna].apply(lambda x: True if 'https://' in str(x) else False ) # Convert to string for checking)


#         if validacion.any():  # Check if any element in 'validacion' is True
#             columnas_a_eliminar.append(columna)
            
#     df.drop(columns=columnas_a_eliminar, inplace=True)
#     df.drop(columns='listing_source', inplace=True)

#     return df


# def reemplazar_vacios_por_nan(df):
#     """
#     Reemplaza los valores vacíos ('' o espacios en blanco) por NaN en las columnas de tipo object de un DataFrame.

#     Args:
#         df (pd.DataFrame): El DataFrame a modificar.

#     Returns:
#         pd.DataFrame: El DataFrame con los valores vacíos reemplazados por NaN.
#     """

#     # Seleccionar las columnas de tipo object
#     columnas_object = df.select_dtypes(include=['object']).columns

#     # Iterar sobre las columnas y reemplazar los valores vacíos
#     for columna in columnas_object:
#         df[columna] = df[columna].replace('', pd.NA)

#     return df

# def label_encode_dataframe(df):
#     le = LabelEncoder()
#     for col in df.columns:
#         if df[col].dtype == 'bool' or df[col].dtype == 'object':
#             # Manejar valores nulos
#             is_null = df[col].isnull()
#             # Convertir a string para asegurar que LabelEncoder funcione
#             values = df[col].fillna('NULL').astype(str)
#             # Aplicar LabelEncoder
#             encoded = le.fit_transform(values)
#             # Restaurar los valores nulos
#             encoded = pd.Series(encoded, index=df.index)
#             encoded[is_null] = np.nan
#             # Asignar los valores codificados de vuelta a la columna
#             df[col] = encoded
#     return df

# def analyze_feature_importance(X, y, top_n=15):
#     mi_scores = mutual_info_classif(X, y)
#     mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
#     mi_scores = mi_scores.sort_values(ascending=False)
#     top_features = mi_scores.index[:top_n]
#     X_reduced = X[top_features]
#     return X_reduced


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import json

class DataProcessor:
    def __init__(self):
        pass

    @staticmethod
    def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
        """Flattens a nested dictionary into a single-level dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(DataProcessor.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                if v and isinstance(v[0], dict):
                    for i, item in enumerate(v):
                        items.extend(DataProcessor.flatten_dict(item, f"{new_key}{sep}{i}", sep=sep).items())
                else:
                    items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def load_and_flatten_data(file_path: str) -> pd.DataFrame:
        """Loads JSON data from a file and flattens it into a Pandas DataFrame."""
        with open(file_path, 'r') as f:
            data = [DataProcessor.flatten_dict(json.loads(line)) for line in f]
        return pd.DataFrame(data)

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans a Pandas DataFrame by removing columns with more than 20% null values,
           columns containing 'id' in their names, and columns with only empty lists."""
        # Autoconsistency: Ensure input is a Pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame.")
        
        # Remove columns with more than 20% null values
        null_percentage = df.isnull().sum() / len(df)
        cols_to_drop = null_percentage[null_percentage > 0.2].index
        df.drop(cols_to_drop, axis=1, inplace=True)

        # Remove columns containing 'id'
        id_cols = df.columns[df.columns.str.contains('id')]
        df.drop(id_cols, axis=1, inplace=True)

        # Remove columns with only empty lists
        cols_to_drop = []
        for col in df.columns:
            validation = df[col].apply(lambda x: True if '[]' in str(x) else False)
            if validation.any():
                cols_to_drop.append(col)
            validation = df[col].apply(lambda x: True if 'http://' in str(x) else False)
            if validation.any():
                cols_to_drop.append(col)
            validation = df[col].apply(lambda x: True if 'https://' in str(x) else False)
            if validation.any():
                cols_to_drop.append(col)
        df.drop(cols_to_drop, axis=1, inplace=True)
        df.drop(['listing_source'], axis=1, inplace=True)

        return df

    def replace_empty_with_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replaces empty strings with NaN in object-type columns of a Pandas DataFrame."""
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            df[col] = df[col].replace('', np.nan)
        return df

    def label_encode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encodes boolean and object-type columns in a Pandas DataFrame."""
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == bool or df[col].dtype == object:
                is_null = df[col].isnull()
                values = df[col].fillna('NULL').astype(str)
                encoded = le.fit_transform(values)
                encoded = pd.Series(encoded, index=df.index)
                encoded[is_null] = np.nan
                df[col] = encoded
        return df

    def feature_importance_analysis(self, X: pd.DataFrame, y: pd.Series, top_n: int = 15) -> pd.DataFrame:
        """Analyzes feature importance using Mutual Information and returns the top N features."""
        mi_scores = mutual_info_classif(X, y)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        top_features = mi_scores.index[:top_n]
        X_reduced = X[top_features]
        return X_reduced