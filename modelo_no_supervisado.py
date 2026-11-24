"""Analisis no supervisado de volcanes: K-Means, DBSCAN y PCA.

Uso:
    pip install -r requirements.txt
    python modelo_no_supervisado.py

El Excel original tiene encabezados reales en la fila 2 (header=1).
No se utilizan Activity Evidence ni Last Known Eruption para formar grupos;
se reservan para describir los grupos despues del clustering.
"""
from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "volcanes_original.xlsx"
RESULTS = ROOT / "resultados_no_supervisado"
RESULTS.mkdir(exist_ok=True)
RANDOM_STATE = 42

NUMERIC = ["Latitude", "Longitude", "Elevation (m)"]
CATEGORICAL = [
    "Volcano Landform",
    "Primary Volcano Type",
    "Tectonic Setting",
    "Dominant Rock Type",
]
FEATURES = NUMERIC + CATEGORICAL


def load_data():
    df = pd.read_excel(DATA, sheet_name="Holocene Volcano List", header=1)
    df.columns = [str(c).strip() for c in df.columns]
    required = FEATURES + ["Volcano Name", "Activity Evidence", "Last Known Eruption"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)

    df = df.drop_duplicates().dropna(subset=NUMERIC).copy()
    return df.reset_index(drop=True)


def prepare_features(df):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformer = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])
    matrix = transformer.fit_transform(df[FEATURES])
    return matrix, transformer


def clustering_metrics(X, labels, ignore_noise=False):
    labels = np.asarray(labels)
    mask = labels != -1 if ignore_noise else np.ones(len(labels), dtype=bool)
    X_eval = X[mask]
    y_eval = labels[mask]
    unique = np.unique(y_eval)
    if len(unique) < 2 or len(unique) >= len(y_eval):
        return {"Silhouette": np.nan, "Calinski_Harabasz": np.nan,
                "Davies_Bouldin": np.nan}
    return {
        "Silhouette": float(silhouette_score(X_eval, y_eval)),
        "Calinski_Harabasz": float(calinski_harabasz_score(X_eval, y_eval)),
        "Davies_Bouldin": float(davies_bouldin_score(X_eval, y_eval)),
    }


def run_kmeans(X, df):
    # Modelo inicial: k=3. Optimizacion: probar k de 2 a 10 y elegir mayor silhouette.
    baseline = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20)
    labels_base = baseline.fit_predict(X)
    base_metrics = clustering_metrics(X, labels_base)

    search_rows = []
    best_model, best_labels, best_score = None, None, -np.inf
    for k in range(2, 11):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X)
        scores = clustering_metrics(X, labels)
        search_rows.append({"k": k, **scores, "Inercia": float(model.inertia_)})
        if np.isfinite(scores["Silhouette"]) and scores["Silhouette"] > best_score:
            best_model, best_labels, best_score = model, labels, scores["Silhouette"]

    if best_model is None:
        raise RuntimeError("No se encontro una configuracion valida para K-Means.")

    optimized_metrics = clustering_metrics(X, best_labels)
    comparison = pd.DataFrame([
        {"Modelo": "K-Means inicial (k=3)", "Clusters": 3, **base_metrics},
        {"Modelo": f"K-Means optimizado (k={best_model.n_clusters})",
         "Clusters": best_model.n_clusters, **optimized_metrics},
    ])
    pd.DataFrame(search_rows).to_csv(RESULTS / "busqueda_kmeans.csv", index=False)
    return best_labels, comparison, best_model.n_clusters


def run_dbscan(X):
    # Se estiman eps candidatos a partir de distancias a vecinos cercanos.
    rows = []
    best = None
    best_score = -np.inf
    for min_samples in [5, 10, 15]:
        neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X)
        distances, _ = neighbors.kneighbors(X)
        kth = np.sort(distances[:, -1])
        eps_candidates = sorted(set(np.round(np.quantile(kth, [0.70, 0.80, 0.90, 0.95]), 3)))
        for eps in eps_candidates:
            model = DBSCAN(eps=float(eps), min_samples=min_samples)
            labels = model.fit_predict(X)
            clusters = len(set(labels) - {-1})
            noise_pct = float(np.mean(labels == -1) * 100)
            scores = clustering_metrics(X, labels, ignore_noise=True)
            row = {"eps": float(eps), "min_samples": min_samples,
                   "Clusters": clusters, "Ruido_pct": noise_pct, **scores}
            rows.append(row)
            score = scores["Silhouette"]
            if clusters >= 2 and np.isfinite(score) and score > best_score:
                best, best_score = (model, labels, row), score

    pd.DataFrame(rows).to_csv(RESULTS / "busqueda_dbscan.csv", index=False)
    if best is None:
        return None, pd.DataFrame(rows), None
    model, labels, best_row = best
    return labels, pd.DataFrame(rows), best_row


def main():
    df = load_data()
    X, transformer = prepare_features(df)

    # PCA se utiliza para representar en 2D la estructura de los datos codificados.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(X)
    pca_df = pd.DataFrame({
        "Componente_principal_1": coordinates[:, 0],
        "Componente_principal_2": coordinates[:, 1],
    })
    pca_df.to_csv(RESULTS / "componentes_pca.csv", index=False)

    kmeans_labels, kmeans_comparison, best_k = run_kmeans(X, df)
    df_results = df.copy()
    df_results["Cluster_KMeans"] = kmeans_labels
    df_results["Componente_principal_1"] = coordinates[:, 0]
    df_results["Componente_principal_2"] = coordinates[:, 1]
    kmeans_comparison.to_csv(RESULTS / "comparacion_kmeans.csv", index=False)

    dbscan_labels, dbscan_search, dbscan_best = run_dbscan(X)
    if dbscan_labels is not None:
        df_results["Cluster_DBSCAN"] = dbscan_labels
        dbscan_metrics = clustering_metrics(X, dbscan_labels, ignore_noise=True)
        dbscan_summary = {
            "eps": dbscan_best["eps"],
            "min_samples": dbscan_best["min_samples"],
            "clusters": int(dbscan_best["Clusters"]),
            "noise_percent": dbscan_best["Ruido_pct"],
            **dbscan_metrics,
        }
    else:
        dbscan_summary = {"mensaje": "No se encontro una configuracion DBSCAN con al menos dos clusters."}

    # Tabla descriptiva: ayuda a interpretar los grupos; Activity Evidence no intervino
    # en el entrenamiento y se agrega solo despues para describirlos.
    group_summary = df_results.groupby("Cluster_KMeans").agg(
        Cantidad_volcanes=("Volcano Name", "count"),
        Elevacion_promedio_m=("Elevation (m)", "mean"),
        Latitud_promedio=("Latitude", "mean"),
        Longitud_promedio=("Longitude", "mean"),
    ).reset_index()
    if "Activity Evidence" in df_results.columns:
        activity = (df_results.groupby("Cluster_KMeans")["Activity Evidence"]
                    .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Sin dato")
                    .rename("Evidencia_actividad_mas_comun").reset_index())
        group_summary = group_summary.merge(activity, on="Cluster_KMeans", how="left")
    group_summary.to_csv(RESULTS / "resumen_clusters.csv", index=False)
    df_results.to_csv(RESULTS / "volcanes_transformados_clusters.csv", index=False, encoding="utf-8-sig")

    # Grafica PCA con etiquetas K-Means.
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=kmeans_labels,
                         cmap="tab10", s=24, alpha=0.75)
    ax.set_title(f"Agrupacion de volcanes con K-Means (k={best_k}) sobre PCA")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    fig.colorbar(scatter, ax=ax, label="Cluster K-Means")
    fig.tight_layout()
    fig.savefig(RESULTS / "clusters_kmeans_pca.png", dpi=160)
    plt.close(fig)

    if dbscan_labels is not None:
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=dbscan_labels,
                             cmap="tab10", s=24, alpha=0.75)
        ax.set_title("Agrupacion de volcanes con DBSCAN sobre PCA (-1 = ruido)")
        ax.set_xlabel("Componente principal 1")
        ax.set_ylabel("Componente principal 2")
        fig.colorbar(scatter, ax=ax, label="Cluster DBSCAN")
        fig.tight_layout()
        fig.savefig(RESULTS / "clusters_dbscan_pca.png", dpi=160)
        plt.close(fig)

    # Varianza explicada de PCA.
    pca_variance = pd.DataFrame({
        "Componente": ["PC1", "PC2"],
        "Varianza_explicada": pca.explained_variance_ratio_,
    })
    pca_variance.to_csv(RESULTS / "varianza_pca.csv", index=False)

    with open(RESULTS / "configuracion_modelos.json", "w", encoding="utf-8") as f:
        json.dump({
            "registros_analizados": int(len(df)),
            "variables_utilizadas": FEATURES,
            "variables_excluidas_del_agrupamiento": ["Activity Evidence", "Last Known Eruption", "Volcano Name"],
            "kmeans_k_optimizado": int(best_k),
            "dbscan_mejor_configuracion": dbscan_summary,
            "pca_varianza_explicada": pca.explained_variance_ratio_.tolist(),
        }, f, ensure_ascii=False, indent=2)

    print(f"Registros analizados: {len(df)}")
    print("\nComparacion K-Means:")
    print(kmeans_comparison.to_string(index=False))
    print("\nMejor configuracion DBSCAN:")
    print(json.dumps(dbscan_summary, ensure_ascii=False, indent=2))
    print("\nVarianza explicada PCA:")
    print(pca_variance.to_string(index=False))
    print(f"\nResultados guardados en: {RESULTS}")


if __name__ == "__main__":
    main()
