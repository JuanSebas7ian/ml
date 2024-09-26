import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
from data_process import DataProcessor
import os
import logging


# Crear el directorio 'logs' si no existe
log_dir = 'Framework/logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# Configuramos el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler = logging.FileHandler('Framework/logs/model.log')
handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    processor = DataProcessor()

    try:
        # Cargar datos desde un archivo JSON
        logger.info('Cargando datos...')
        df = processor.load_and_flatten_data("Framework/data/MLA_100k_checked_v3.jsonlines")
        
        # Procesar datos
        logger.info('Procesando datos...')
        df['price_ratio'] = df['base_price'] / df['price']
        df['sold_ratio'] = df['sold_quantity'] / df['initial_quantity']
        df['is_warranty'] = df['warranty'].notna().astype(int)
        df['listing_duration'] = df['stop_time'] - df['start_time']
        df['price_per_quantity'] = df['price'] / df['available_quantity']
        df['price_ratio'] = df['price_ratio'].clip(0, 10)
        df['sold_ratio'] = df['sold_ratio'].clip(0, 1)

        seller_avg_price = df.groupby('seller_id')['price'].mean()
        df['seller_avg_price'] = df['seller_id'].map(seller_avg_price)

        seller_item_count = df['seller_id'].value_counts()
        df['seller_item_count'] = df['seller_id'].map(seller_item_count)

        seller_condition_ratio = df.groupby('seller_id')['condition'].apply(lambda x: (x == 'new').mean())
        df['seller_new_ratio'] = df['seller_id'].map(seller_condition_ratio)
        df['date_created'] = pd.to_datetime(df['date_created'])
        df['day_of_week'] = df['date_created'].dt.dayofweek
        df['month'] = df['date_created'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Limpiar el dataframe
        logger.info('Limpiando dataframe...')
        data_limpio = processor.clean_dataframe(df)

        # Reemplazar valores vacíos con NaN
        logger.info('Reemplazando valores vacíos...')
        data_limpio = processor.replace_empty_with_nan(data_limpio)
        data_limpio=data_limpio.query("status=='active'")
        data_limpio=data_limpio.query("buying_mode=='buy_it_now'")

        data_limpio.drop(['status','buying_mode','seller_address_country_name', 'accepts_mercadopago','international_delivery_mode','pictures_0_quality','international_delivery_mode'], axis=1, inplace=True)

        # Codificar etiquetas en columnas boolean y object-type
        logger.info('Codificando etiquetas...')
        data_limpio = processor.label_encode_dataframe(data_limpio)
        data_limpio.drop(columns=[ 'base_price','initial_quantity','listing_duration','seller_avg_price','price_per_quantity'], axis=1, inplace=True)

        X = data_limpio.drop('condition', axis=1)
        y = data_limpio['condition']

        # Asegurar que las características sean numéricas
        logger.info('Convirtiendo características a numéricas...')
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Analizar importancia de características
        logger.info('Analizando importancia de características...')
        X_reduced = processor.feature_importance_analysis(X, y)
        X_reduced.drop(columns=['date_created','start_time'], axis=1, inplace=True)

        
        # Crear el directorio 'data' si no existe
        data_dir = 'Framework/data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Guardar resultados en archivos CSV
        logger.info('Guardando resultados...')
        X_reduced.to_csv('Framework/data/X_reduced.csv', index=False)
        y.to_csv('Framework/data/y.csv', index=False)

    except Exception as e:
        logger.error(f'Ocurrió un error: {str(e)}')
    finally:
        logger.info('Fin del proceso.')

if __name__ == '__main__':
    current_dir = os.getcwd()
    log_dir = os.path.join(current_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(os.path.join(log_dir,'model.log'))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    main()