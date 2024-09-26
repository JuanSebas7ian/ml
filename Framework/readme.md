
# Proyecto: Clasificación de Artículos Nuevos y Usados con Random Forest

## Menú

- [Proyecto: Clasificación de Artículos Nuevos y Usados con Random Forest](#proyecto-clasificación-de-artículos-nuevos-y-usados-con-random-forest)
  - [Menú](#menú)
  - [Introducción](#introducción)
  - [Justificación](#justificación)
  - [Descripción de la Estructura del Proyecto](#descripción-de-la-estructura-del-proyecto)
    - [`Framework/`](#framework)
    - [`data/`](#data)
    - [`logs/`](#logs)
    - [`model/`](#model)
  - [Métodos y Clases Implementados](#métodos-y-clases-implementados)
    - [**data\_process.py**](#data_processpy)
    - [**classifier.py**](#classifierpy)
    - [**steps.py**](#stepspy)
  - [Entrenamiento y Evaluación del Modelo](#entrenamiento-y-evaluación-del-modelo)
  - [Resultados](#resultados)
  - [Cómo Ejecutar el Proyecto](#cómo-ejecutar-el-proyecto)

## Introducción

Este proyecto tiene como objetivo entrenar un modelo de clasificación para predecir si un artículo es **nuevo** o **usado**, utilizando características extraídas de un conjunto de datos de artículos de un mercado en línea.

Se ha implementado un **Random Forest Classifier**, que es un conjunto de árboles de decisión, lo cual aumenta la robustez y generalización del modelo. Además, se utilizan varias métricas para evaluar el rendimiento, como precisión, sensibilidad, especificidad, F1-Score y AUC-ROC.

## Justificación

La clasificación precisa de los artículos en categorías como "nuevos" o "usados" es crucial para los vendedores, ya que mejora la experiencia de usuario al reducir el error en la clasificación de productos. También ayuda a los compradores a tomar decisiones informadas y a filtrar mejor los artículos que desean adquirir.

La elección de un modelo de **Random Forest** es ideal debido a su capacidad para manejar grandes cantidades de características y su flexibilidad para modelos no lineales. Además, Random Forest maneja eficientemente datos faltantes y es menos propenso al sobreajuste.

## Descripción de la Estructura del Proyecto

### `Framework/`
Contiene los scripts principales del proyecto:
- `classifier.py`: Implementa el modelo Random Forest y las métricas de evaluación.
- `data_process.py`: Contiene funciones para cargar, limpiar y preprocesar los datos.
- `orchestor.py`: Orquestador principal para la ejecución de las fases del proyecto.
- `steps.py`: Ejecuta los pasos principales del flujo de trabajo.

### `data/`
Carpeta donde se almacenan los conjuntos de datos y archivos procesados:
- `MLA_100k_checked_v3.jsonlines`: Datos crudos del problema.
- `X_reduced.csv`: Conjunto de datos preprocesado y reducido para el modelo.

### `logs/`
Contiene los archivos de registro:
- `model_classifier.log`: Registro de eventos clave durante el entrenamiento del modelo.
- `model.log`: Registro adicional que incluye detalles sobre el rendimiento del modelo.

### `model/`
Contiene el modelo entrenado y el escalador:
- `random_forest_model.joblib`: El modelo de Random Forest entrenado, listo para ser cargado y usado en inferencia.

## Métodos y Clases Implementados

### **data_process.py**
- **`load_and_flatten_data(filepath)`**: Carga el archivo `jsonlines` y lo convierte en un DataFrame plano para su posterior uso.
- **`preprocess_data(df)`**: Realiza el preprocesamiento del DataFrame, como la imputación de valores faltantes, codificación de variables categóricas, y escalado de características.

### **classifier.py**
- **`train_random_forest(X_train, y_train)`**: Entrena el modelo Random Forest con los datos de entrenamiento.
- **`evaluate_model(model, X_test, y_test)`**: Calcula las métricas de rendimiento: Precisión, Sensibilidad (Recall), Especificidad (Precision), F1-Score y AUC-ROC.
- **`save_model(model, filepath)`**: Guarda el modelo entrenado en un archivo `joblib`.

### **steps.py**
- **`main()`**: Controla el flujo del proyecto, llamando a las funciones de preprocesamiento, entrenamiento y evaluación.
- **`orchestrate_steps()`**: Llama al orquestador (`orchestor.py`) para ejecutar las diferentes fases del proyecto.

## Entrenamiento y Evaluación del Modelo

1. **Preprocesamiento**: Los datos se cargan y se procesan para eliminar valores nulos y codificar variables categóricas. 
   
2. **División del Conjunto de Datos**: El conjunto de datos se divide en conjunto de entrenamiento y prueba en una proporción 80/20.

3. **Entrenamiento**: Se entrena un modelo de Random Forest con los datos de entrenamiento preprocesados.

4. **Evaluación**: Se evalúa el modelo usando el conjunto de prueba y se calculan las métricas mencionadas.

## Resultados

El modelo entrenado ha obtenido los siguientes resultados durante la fase de evaluación:

- **Precisión (accuracy):** 96.17%
- **Sensibilidad (recall):** 96.84%
- **Especificidad (precision):** 95.60%
- **F1-Score:** 96.22%
- **AUC-ROC:** 99.39%
- **Train Accuracy:** 100.00%
- **Test Accuracy:** 96.17%

Estos resultados muestran que el modelo tiene un rendimiento muy alto, con una excelente capacidad para generalizar en los datos de prueba.

## Cómo Ejecutar el Proyecto

1. **Prepara el entorno:**
   - Clona este repositorio en tu máquina local.
   - Asegúrate de tener todas las dependencias instaladas (ver `requirements.txt`).

2. **Ejecuta los pasos:**
   Para iniciar el flujo de trabajo, ejecuta el siguiente comando:

   ```bash
   python steps.py
