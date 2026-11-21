"""
Proyecto de aprendizaje supervisado: datos de volcanes.
Clasificación: Activity Evidence.
Regresión: Elevation (m).

Uso:
    pip install -r requirements.txt
    python modelo_volcanes.py

El archivo Excel original tiene dos filas iniciales de título/metadatos;
por eso se lee la hoja principal con header=1.
"""
from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "volcanes_original.xlsx"
RESULTS = ROOT / "resultados"
RESULTS.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

def load_and_clean():
    # La primera fila contiene títulos del reporte; la segunda contiene encabezados reales.
    df = pd.read_excel(DATA, sheet_name="Holocene Volcano List", header=1)
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "Country", "Volcanic Region Group", "Volcanic Region",
        "Volcano Landform", "Primary Volcano Type", "Activity Evidence",
        "Latitude", "Longitude", "Elevation (m)", "Tectonic Setting",
        "Dominant Rock Type"
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas en el Excel: {missing_cols}")

    # Convertir coordenadas y elevación a valores numéricos.
    for col in ["Latitude", "Longitude", "Elevation (m)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Eliminar duplicados exactos y registros sin etiqueta/objetivo.
    df = df.drop_duplicates()
    df = df.dropna(subset=["Activity Evidence", "Elevation (m)"]).copy()

    # Normalizar espacios en columnas de texto; los faltantes se conservan
    # para imputarlos dentro del pipeline y evitar fuga de información.
    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

    df.to_csv(RESULTS / "volcanes_limpios.csv", index=False, encoding="utf-8-sig")
    return df

def make_preprocessor(X):
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])

def classification(df):
    features = [
        "Country", "Volcanic Region Group", "Volcanic Region",
        "Volcano Landform", "Primary Volcano Type", "Latitude",
        "Longitude", "Elevation (m)", "Tectonic Setting",
        "Dominant Rock Type"
    ]
    X, y = df[features], df["Activity Evidence"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Regresión logística": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1
        )
    }
    rows, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([("prep", make_preprocessor(X_train)), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "Modelo": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
            "Recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
            "F1_macro": f1_score(y_test, pred, average="macro", zero_division=0)
        })
        fitted[name] = pipe

    # Búsqueda de parámetros para Random Forest. La validación ocurre solo
    # dentro del conjunto de entrenamiento.
    rf_pipe = Pipeline([
        ("prep", make_preprocessor(X_train)),
        ("model", RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ))
    ])
    grid = {
        "model__n_estimators": [150, 250],
        "model__max_depth": [None, 15],
        "model__min_samples_leaf": [1, 2]
    }
    search = GridSearchCV(
        rf_pipe, grid, cv=3, scoring="f1_macro", n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    pred = search.predict(X_test)
    rows.append({
        "Modelo": "Random Forest optimizado",
        "Accuracy": accuracy_score(y_test, pred),
        "Precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "Recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
        "F1_macro": f1_score(y_test, pred, average="macro", zero_division=0)
    })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metricas_clasificacion.csv", index=False)
    with open(RESULTS / "mejores_parametros_clasificacion.json", "w", encoding="utf-8") as f:
        json.dump(search.best_params_, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(10, 5))
    metrics.set_index("Modelo")[["Accuracy", "Precision_macro", "Recall_macro", "F1_macro"]].plot(
        kind="bar", ax=ax
    )
    ax.set_title("Comparación de modelos de clasificación")
    ax.set_ylabel("Métrica (0 a 1)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(RESULTS / "comparacion_clasificacion.png", dpi=160)
    plt.close(fig)

    ConfusionMatrixDisplay.from_estimator(search.best_estimator_, X_test, y_test, xticks_rotation=45)
    plt.title("Matriz de confusión - Random Forest optimizado")
    plt.tight_layout()
    plt.savefig(RESULTS / "matriz_confusion.png", dpi=160)
    plt.close()
    return metrics

def regression(df):
    features = [
        "Country", "Volcanic Region Group", "Volcanic Region",
        "Volcano Landform", "Primary Volcano Type", "Activity Evidence",
        "Latitude", "Longitude", "Tectonic Setting", "Dominant Rock Type"
    ]
    X, y = df[features], df["Elevation (m)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    models = {
        "Regresión lineal": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        )
    }
    rows = []
    for name, model in models.items():
        pipe = Pipeline([("prep", make_preprocessor(X_train)), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "Modelo": name,
            "MAE_m": mean_absolute_error(y_test, pred),
            "RMSE_m": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred)
        })

    rf_pipe = Pipeline([
        ("prep", make_preprocessor(X_train)),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
    ])
    grid = {
        "model__n_estimators": [150, 250],
        "model__max_depth": [None, 15],
        "model__min_samples_leaf": [1, 2]
    }
    search = GridSearchCV(
        rf_pipe, grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    pred = search.predict(X_test)
    rows.append({
        "Modelo": "Random Forest optimizado",
        "MAE_m": mean_absolute_error(y_test, pred),
        "RMSE_m": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred)
    })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metricas_regresion.csv", index=False)
    with open(RESULTS / "mejores_parametros_regresion.json", "w", encoding="utf-8") as f:
        json.dump(search.best_params_, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(9, 5))
    metrics.set_index("Modelo")[["MAE_m", "RMSE_m"]].plot(kind="bar", ax=ax)
    ax.set_title("Comparación de errores de regresión")
    ax.set_ylabel("Error en metros (menor es mejor)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(RESULTS / "errores_regresion.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(metrics["Modelo"], metrics["R2"])
    ax.set_title("Comparación del coeficiente R²")
    ax.set_ylabel("R² (mayor es mejor)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(RESULTS / "r2_regresion.png", dpi=160)
    plt.close(fig)
    return metrics

def main():
    df = load_and_clean()
    print(f"Registros limpios: {len(df)}")
    print("\nClasificación:")
    print(classification(df).to_string(index=False))
    print("\nRegresión:")
    print(regression(df).to_string(index=False))
    print(f"\nArchivos generados en: {RESULTS}")

if __name__ == "__main__":
    main()
