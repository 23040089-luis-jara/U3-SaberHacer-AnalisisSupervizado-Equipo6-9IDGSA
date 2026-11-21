# Modelos supervisados con datos de volcanes

## Objetivo
Desarrollar un modelo de clasificación para predecir `Activity Evidence` y un modelo de regresión para estimar `Elevation (m)`.

## Estructura
- `data/volcanes_original.xlsx`: archivo original recibido, sin modificaciones.
- `data/volcanes_limpios.csv`: se genera al ejecutar el proceso de limpieza.
- `modelo_volcanes.py`: código fuente completo de limpieza, entrenamiento, optimización y evaluación.
- `notebooks/proceso_modelos_volcanes.ipynb`: notebook para ejecutar el proceso.
- `resultados/`: métricas, parámetros y gráficas generadas.

## Requisitos
Se recomienda Python 3.10 o superior.

```bash
pip install -r requirements.txt
```

## Ejecutar
Desde la carpeta raíz del proyecto:

```bash
python modelo_volcanes.py
```

También se puede abrir `notebooks/proceso_modelos_volcanes.ipynb` en Jupyter Notebook o VS Code y ejecutar sus celdas.

## Preparación de datos
El archivo original contiene una fila de título antes de los encabezados reales. Por ello, el código lee la hoja `Holocene Volcano List` con `header=1`. Se eliminan duplicados exactos y filas sin variable objetivo o elevación. Los faltantes de variables predictoras se imputan dentro de los pipelines, ajustando los imputadores solo con los datos de entrenamiento.

## Modelos
- Clasificación: Regresión logística, Random Forest y Random Forest optimizado.
- Regresión: Regresión lineal, Random Forest y Random Forest optimizado.
- División: 80 % entrenamiento y 20 % prueba, semilla 42.
- Clasificación: división estratificada y métricas Accuracy, Precision macro, Recall macro y F1 macro.
- Regresión: MAE, RMSE y R².
- Optimización: GridSearchCV con validación cruzada de 3 particiones.

## Nota importante
Los resultados se calculan al ejecutar el código. No se deben copiar métricas de una ejecución anterior si se modifican los datos, las variables o los parámetros. Las predicciones describen patrones de esta base de datos; no constituyen pronósticos de erupciones ni evaluaciones de riesgo volcánico.
