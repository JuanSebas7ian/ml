import os
import sys
import logging
import pandas as pd
from orchestor import main as data_processing_main
from classifier import RandomForestModel

# Configuramos el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_menu():
    print("\nSeleccione una opción:")
    print("1. Procesar y limpiar datos")
    print("2. Entrenar y evaluar modelo Random Forest")
    print("3. Verificar sobreajuste del modelo")
    print("4. Cargar modelo guardado y hacer predicciones")
    print("5. Salir")

def process_data():
    logger.info("Iniciando proceso de carga y limpieza de datos...")
    data_processing_main()
    logger.info("Datos procesados y almacenados.")

def train_and_evaluate_model():
    logger.info("Iniciando entrenamiento del modelo Random Forest...")
    rfm = RandomForestModel()
    try:
        X_reduced = pd.read_csv('data/X_reduced.csv')
        y = pd.read_csv('data/y.csv')
        model, scaler, train_acc, test_acc, X_test, y_test, X_test_scaled = rfm.train_and_evaluate(X_reduced, y)
        rfm.save_model(model, scaler)
        logger.info(f"Entrenamiento completado. Precisión en entrenamiento: {train_acc:.4f}, Precisión en prueba: {test_acc:.4f}")
    except Exception as e:
        logger.error(f"Ocurrió un error al entrenar el modelo: {str(e)}")

def check_model_overfitting():
    logger.info("Verificando sobreajuste del modelo...")
    rfm = RandomForestModel()
    try:
        X_reduced = pd.read_csv('data/X_reduced.csv')
        y = pd.read_csv('data/y.csv')
        is_overfitting = rfm.check_overfitting(X_reduced, y)
        if is_overfitting:
            logger.warning("El modelo muestra signos de sobreajuste.")
        else:
            logger.info("El modelo no muestra signos de sobreajuste.")
    except Exception as e:
        logger.error(f"Ocurrió un error al verificar el sobreajuste: {str(e)}")

def load_model_and_predict():
    logger.info("Cargando modelo y haciendo predicciones...")
    rfm = RandomForestModel()
    try:
        X_reduced = pd.read_csv('data/X_reduced.csv')  # Nuevos datos para predecir
        predictions = rfm.load_model_and_predict(X_new=X_reduced)
        logger.info(f"Predicciones realizadas: {predictions}")
    except Exception as e:
        logger.error(f"Ocurrió un error al cargar el modelo o hacer predicciones: {str(e)}")

def main():
    while True:
        print_menu()
        choice = input("\nIngrese su elección: ")
        
        if choice == '1':
            process_data()
        elif choice == '2':
            train_and_evaluate_model()
        elif choice == '3':
            check_model_overfitting()
        elif choice == '4':
            load_model_and_predict()
        elif choice == '5':
            logger.info("Saliendo del programa.")
            sys.exit()
        else:
            print("Opción no válida, por favor intente nuevamente.")

if __name__ == '__main__':
    main()
