# Analisis no supervisado de volcanes

Este complemento del proyecto de aprendizaje supervisado aplica tecnicas de clustering y reduccion de dimensionalidad al inventario de volcanes.

## Algoritmos
- K-Means: agrupamiento inicial con k=3 y busqueda de k entre 2 y 10 usando Silhouette Score.
- DBSCAN: agrupamiento basado en densidad; se ajustan `eps` y `min_samples`. Los registros etiquetados como -1 se consideran ruido.
- PCA: reduccion a dos componentes para visualizar la estructura de los datos.

## Variables
Se utilizan `Latitude`, `Longitude`, `Elevation (m)`, `Volcano Landform`, `Primary Volcano Type`, `Tectonic Setting` y `Dominant Rock Type`. Las variables `Activity Evidence`, `Last Known Eruption` y `Volcano Name` no participan en el agrupamiento. La evidencia de actividad se agrega despues unicamente para describir los clusters.

## Ejecucion
```bash
pip install -r requirements.txt
python modelo_no_supervisado.py
```

Tambien se puede abrir `notebooks/proceso_no_supervisado_volcanes.ipynb` y ejecutar sus celdas.

## Salidas
Los resultados se guardan en `resultados_no_supervisado/`: comparacion de K-Means, busquedas de parametros, configuracion elegida, tabla resumen de clusters, dataset con etiquetas, componentes PCA, varianza explicada y graficas.

## Resultados de la ejecucion de referencia
Con 1,214 registros, K-Means con k=3 obtuvo Silhouette=0.177966, Calinski-Harabasz=231.296107 y Davies-Bouldin=1.928485. La busqueda entre k=2 y k=10 selecciono k=8 por su mayor Silhouette (0.183657), con Calinski-Harabasz=157.205150 y Davies-Bouldin=1.743233. DBSCAN selecciono eps=1.887 y min_samples=15: encontro 2 clusters y marco 14.25% de los registros como ruido. PCA explico 22.63% de la varianza con PC1 y 17.93% con PC2.

Estos valores corresponden a la ejecucion de referencia y pueden variar si cambia el dataset o las versiones de las bibliotecas.
