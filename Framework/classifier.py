import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import os


# Crear el directorio 'logs' si no existe
log_dir = 'Framework/logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

class RandomForestModel:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.FileHandler('Framework/logs/model_clasifier.log')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def train_and_evaluate(self, X, y, n_estimators=100, test_size=0.2, random_state=42):
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train the model
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate accuracies
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)

            # Aquí agregas las nuevas métricas (desde aquí hasta el final del bloque)
            cm = confusion_matrix(y_test, y_pred_test)
            tn, fp, fn, tp = cm.ravel()
            self.logger.info(f"Matriz de confusión:\n{cm}")
            self.logger.info(f"Precisión (accuracy): {(tp + tn) / (tn + fp + fn + tp):.4f}")
            self.logger.info(f"Sensibilidad (recall): {tp / (fn + tp):.4f}")
            self.logger.info(f"Especificidad (precision): {tn / (fp + tn):.4f}")
            self.logger.info(f"F1-score: {2 * ((tp / (fn + tp)) * (tn / (fp + tn))) / ((tp / (fn + tp)) + (tn / (fp + tn))):.4f}")

            fpr, tpr, thresholds = roc_curve(y_test, [x[1] for x in model.predict_proba(X_test_scaled)])
            auc_value = auc(fpr, tpr)
            self.logger.info(f"AUC-ROC: {auc_value:.4f}")

            self.logger.info(f'Train accuracy: {train_accuracy:.4f}')
            self.logger.info(f'Test accuracy: {test_accuracy:.4f}')

            return model, scaler, train_accuracy, test_accuracy, X_test, y_test, X_test_scaled
        
        except Exception as e:
            self.logger.error(f'Ocurrió un error: {str(e)}')
            return None
    
    def check_overfitting(self, X, y, n_runs=5, threshold=0.05):
        try:
            train_accuracies = []
            test_accuracies = []
            cv_scores = []
            
            for _ in range(n_runs):
                _, _, train_acc, test_acc, _, _, _ = self.train_and_evaluate(X, y)
                train_accuracies.append(train_acc)
                test_accuracies.append(test_acc)
                
                # Perform cross-validation
                cv_score = np.mean(cross_val_score(RandomForestClassifier(), X, y, cv=5))
                cv_scores.append(cv_score)
            
            avg_train_acc = np.mean(train_accuracies)
            avg_test_acc = np.mean(test_accuracies)
            avg_cv_score = np.mean(cv_scores)
            
            self.logger.info(f'Promedio de precisión de entrenamiento: {avg_train_acc:.4f}')
            self.logger.info(f'Promedio de precisión de prueba: {avg_test_acc:.4f}')
            self.logger.info(f'Promedio de puntuación de validación cruzada: {avg_cv_score:.4f}')
            
            # Comprobación de sobreajuste
            if avg_train_acc - avg_test_acc > threshold:
                self.logger.warning('El modelo puede estar sobreajustándose.')
                return True
            else:
                self.logger.info('El modelo no muestra signos de sobreajuste.')
                return False
        
        except Exception as e:
            self.logger.error(f'Ocurrió un error: {str(e)}')
            return None
    
    def save_model(self, model, scaler, filename='Framework/model/random_forest_model.joblib'):
        try:
            joblib.dump({'model': model,'scaler': scaler}, filename)
            self.logger.info(f'Modelo y escalador guardados en {filename}')
        
        except Exception as e:
            self.logger.error(f'Ocurrió un error: {str(e)}')
    
    def load_model_and_predict(self, filename='Framework/model/random_forest_model.joblib', X_new=None):
        try:
            loaded = joblib.load(filename)
            model = loaded['model']
            scaler = loaded['scaler']
            
            self.logger.info(f'Modelo y escalador cargados desde {filename}')
            
            if X_new is not None:
                X_new_scaled = scaler.transform(X_new)
                predicciones = model.predict(X_new_scaled)
                return predicciones
            else:
                return model, scaler
        
        except Exception as e:
            self.logger.error(f'Ocurrió un error: {str(e)}')
            return None

def main():
    rfm = RandomForestModel()
    X_reduced = pd.read_csv('Framework/data/X_reduced.csv')
    y = pd.read_csv('Framework/data/y.csv')
    
    modelo, escalador, precision_entrenamiento, precision_prueba, X_prueba, y_prueba, X_prueba_escala = rfm.train_and_evaluate(X_reduced, y)
    rfm.save_model(modelo, escalador)

if __name__ == '__main__':
    main()