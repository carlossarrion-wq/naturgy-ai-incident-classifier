#!/usr/bin/env python3
"""
Demostración de extracción y análisis de centroides del modelo Naturgy
"""

import pandas as pd
import numpy as np
import pickle
import json
import sys
import os

# Importar las clases necesarias
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from naturgy_ai_incident_classifier_fixed import NaturgyIncidentClassifier, TextPreprocessor, IncidentClusterer, PredictiveClassifier, EntityExtractor

def explain_centroids():
    """Explica y demuestra la extracción de centroides"""
    
    print("🎯 EXPLICACIÓN DE CENTROIDES EN EL SISTEMA NATURGY")
    print("=" * 60)
    
    # Cargar modelo entrenado
    try:
        with open('naturgy_incident_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        clusterer = model_data['clusterer']
        clustering_model = clusterer.clustering_model
        vectorizer = clusterer.vectorizer
        
        print("✅ Modelo cargado exitosamente")
        
    except FileNotFoundError:
        print("❌ Modelo no encontrado. Ejecuta primero el entrenamiento.")
        return
    
    # PASO 1: EXTRAER CENTROIDES
    print("\n📊 PASO 1: EXTRACCIÓN DE CENTROIDES")
    print("-" * 40)
    
    centroids = clustering_model.cluster_centers_
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"🔢 Número de centroides: {len(centroids)}")
    print(f"📏 Dimensiones por centroide: {len(centroids[0])}")
    print(f"📝 Palabras en vocabulario: {len(feature_names)}")
    
    # PASO 2: ANALIZAR UN CENTROIDE ESPECÍFICO
    print("\n🔍 PASO 2: ANÁLISIS DEL CENTROIDE DEL CLUSTER 0")
    print("-" * 40)
    
    # Tomar el primer centroide como ejemplo
    centroid_0 = centroids[0]
    
    print(f"📊 Forma del centroide: {centroid_0.shape}")
    print(f"📈 Norma euclidiana: {np.linalg.norm(centroid_0):.3f}")
    print(f"🎯 Valores no-cero: {np.count_nonzero(centroid_0)}/{len(centroid_0)}")
    
    # PASO 3: PALABRAS MÁS IMPORTANTES
    print("\n🏷️ PASO 3: PALABRAS MÁS IMPORTANTES DEL CLUSTER 0")
    print("-" * 40)
    
    # Obtener índices de las palabras con mayor peso
    top_indices = np.argsort(centroid_0)[-15:][::-1]
    
    print("Top 15 palabras más características:")
    for i, idx in enumerate(top_indices, 1):
        if centroid_0[idx] > 0:
            palabra = feature_names[idx]
            peso = centroid_0[idx]
            importancia = peso / np.max(centroid_0) * 100
            print(f"  {i:2d}. {palabra:20s} → {peso:.4f} ({importancia:.1f}%)")
    
    # PASO 4: COMPARAR CENTROIDES
    print("\n🔄 PASO 4: DISTANCIAS ENTRE CENTROIDES")
    print("-" * 40)
    
    from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
    
    # Calcular distancias euclidianas
    distances = euclidean_distances(centroids)
    
    # Encontrar los pares más similares y más diferentes
    upper_indices = np.triu_indices_from(distances, k=1)
    upper_distances = distances[upper_indices]
    
    min_dist_idx = np.argmin(upper_distances)
    max_dist_idx = np.argmax(upper_distances)
    
    min_pair = (upper_indices[0][min_dist_idx], upper_indices[1][min_dist_idx])
    max_pair = (upper_indices[0][max_dist_idx], upper_indices[1][max_dist_idx])
    
    print(f"🔗 Clusters más similares: {min_pair[0]} ↔ {min_pair[1]} (distancia: {upper_distances[min_dist_idx]:.3f})")
    print(f"🔀 Clusters más diferentes: {max_pair[0]} ↔ {max_pair[1]} (distancia: {upper_distances[max_dist_idx]:.3f})")
    print(f"📊 Distancia promedio: {np.mean(upper_distances):.3f}")
    
    # PASO 5: INTERPRETAR SEMÁNTICAMENTE
    print("\n🧠 PASO 5: INTERPRETACIÓN SEMÁNTICA DE CENTROIDES")
    print("-" * 40)
    
    # Analizar algunos centroides
    for cluster_id in [0, 1, 2]:
        centroid = centroids[cluster_id]
        top_indices = np.argsort(centroid)[-5:][::-1]
        top_words = [feature_names[idx] for idx in top_indices if centroid[idx] > 0]
        
        print(f"Cluster {cluster_id}: {' + '.join(top_words[:3])}")
    
    # PASO 6: GENERAR REPORTE DETALLADO
    print(f"\n📋 PASO 6: GENERANDO REPORTE DETALLADO")
    print("-" * 40)
    
    report = {
        'metadata': {
            'num_clusters': len(centroids),
            'num_features': len(feature_names),
            'fecha_analisis': pd.Timestamp.now().isoformat()
        },
        'centroids_analysis': []
    }
    
    for cluster_id, centroid in enumerate(centroids):
        # Top palabras
        top_indices = np.argsort(centroid)[-10:][::-1]
        top_features = []
        
        for idx in top_indices:
            if centroid[idx] > 0:
                top_features.append({
                    'palabra': feature_names[idx],
                    'peso_tfidf': float(centroid[idx]),
                    'importancia_relativa': float(centroid[idx] / np.max(centroid))
                })
        
        cluster_analysis = {
            'cluster_id': cluster_id,
            'centroid_properties': {
                'norma_euclidiana': float(np.linalg.norm(centroid)),
                'features_activos': int(np.count_nonzero(centroid)),
                'densidad': float(np.count_nonzero(centroid) / len(centroid))
            },
            'top_features': top_features[:5]
        }
        
        report['centroids_analysis'].append(cluster_analysis)
    
    # Guardar reporte
    with open('centroids_analysis_naturgy.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("✅ Reporte detallado guardado en 'centroids_analysis_naturgy.json'")
    
    print("\n🎯 RESUMEN: ¿QUÉ SON LOS CENTROIDES?")
    print("=" * 60)
    print("Los centroides son vectores de 8,000 dimensiones que representan")
    print("el 'perfil promedio' de cada tipo de incidencia.")
    print()
    print("🔢 Cada dimensión = peso de una palabra específica")
    print("📊 Peso alto = palabra muy característica del cluster")
    print("🎯 Se usan para clasificar nuevas incidencias")
    print("📏 Distancia a centroide = similitud a ese tipo")


if __name__ == "__main__":
    explain_centroids()
