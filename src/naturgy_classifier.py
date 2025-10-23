#!/usr/bin/env python3
# Naturgy AI Incident Classifier - Complete Pipeline
"""
Sistema completo de clasificación automática de incidencias para Naturgy Delta
Incluye preprocesamiento, clustering, y modelos predictivos para triaje automático
"""

import pandas as pd
import numpy as np
import re
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# NLP Libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import SnowballStemmer
    NLTK_AVAILABLE = True
except ImportError:
    print("NLTK not installed. Some features may be limited.")
    NLTK_AVAILABLE = False

class NaturgyIncidentClassifier:
    """Pipeline completo para clasificación automática de incidencias"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.preprocessor = TextPreprocessor(self.config)
        self.clusterer = IncidentClusterer(self.config)
        self.classifier = PredictiveClassifier(self.config)
        self.entity_extractor = EntityExtractor()
        
        self.is_trained = False
        self.incident_types = {}
        self.model_metrics = {}
        
    def _default_config(self) -> Dict:
        """Configuración por defecto del sistema"""
        return {
            'max_clusters': 50,
            'min_cluster_size': 20,
            'tfidf_max_features': 8000,
            'tfidf_min_df': 3,
            'tfidf_max_df': 0.7,
            'random_state': 42,
            'use_llm': False,
            'model_type': 'random_forest',
            'cv_folds': 5,
            'use_hierarchical': True,
            'silhouette_threshold': 0.1
        }
    
    def train_pipeline(self, data_path: str) -> Dict[str, Any]:
        """Entrena el pipeline completo"""
        print("🚀 Iniciando entrenamiento del pipeline de clasificación...")
        
        # 1. Cargar y limpiar datos
        print("📊 Cargando datos...")
        df = self._load_data(data_path)
        
        # 2. Preprocesamiento
        print("🧹 Preprocesando texto...")
        df_processed = self.preprocessor.process_dataframe(df)
        
        # 3. Extracción de entidades
        print("🔍 Extrayendo entidades...")
        df_processed = self.entity_extractor.extract_entities(df_processed)
        
        # 4. Clustering inicial
        print("🎯 Realizando clustering inicial...")
        cluster_results = self.clusterer.cluster_incidents(df_processed)
        
        # 5. Entrenamiento del modelo predictivo
        print("🤖 Entrenando modelo predictivo...")
        model_results = self.classifier.train_model(
            df_processed, cluster_results['labels']
        )
        
        # 6. Generar tipos de incidencia
        print("📋 Generando definiciones de tipos...")
        self.incident_types = self._generate_incident_types(
            df_processed, cluster_results
        )
        
        # 7. Evaluación del sistema
        print("📈 Evaluando performance...")
        self.model_metrics = self._evaluate_system(
            df_processed, cluster_results, model_results
        )
        
        self.is_trained = True
        
        # 8. Guardar modelo
        self._save_model()
        
        return {
            'incident_types': self.incident_types,
            'metrics': self.model_metrics,
            'cluster_results': cluster_results,
            'model_results': model_results
        }
    
    def classify_incident(self, incident_text: str, 
                         additional_fields: Optional[Dict] = None) -> Dict[str, Any]:
        """Clasifica una nueva incidencia"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
            
        # Crear DataFrame temporal para procesamiento
        temp_df = pd.DataFrame({
            'Resumen': [incident_text],
            'Notas': [additional_fields.get('notas', '') if additional_fields else ''],
            'Tipo de ticket': [additional_fields.get('tipo_ticket', '') if additional_fields else '']
        })
        
        # Procesar texto
        processed_df = self.preprocessor.process_dataframe(temp_df)
        processed_df = self.entity_extractor.extract_entities(processed_df)
        
        # Clasificar
        prediction = self.classifier.predict(processed_df)
        
        # Obtener información del tipo
        incident_type = self.incident_types.get(prediction[0], {})
        
        return {
            'predicted_type': prediction[0],
            'confidence': prediction[1] if len(prediction) > 1 else 0.0,
            'type_info': incident_type,
            'extracted_entities': processed_df['entities'].iloc[0] if 'entities' in processed_df.columns else {},
            'processed_text': processed_df['combined_text'].iloc[0] if 'combined_text' in processed_df.columns else ''
        }
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Carga datos desde Excel"""
        try:
            df = pd.read_excel(data_path)
            print(f"✅ Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")
            return df
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            raise
    
    def _generate_incident_types(self, df: pd.DataFrame, 
                               cluster_results: Dict) -> Dict[str, Dict]:
        """Genera definiciones de tipos de incidencia completamente automáticamente"""
        types = {}
        labels = cluster_results['labels']
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Ruido en clustering
                continue
                
            cluster_mask = labels == cluster_id
            cluster_df = df[cluster_mask]
            
            # Generar nombre y descripción COMPLETAMENTE automáticos
            type_info = {
                'nombre': self._generate_automatic_type_name(cluster_df, cluster_id),
                'descripcion': self._generate_type_description(cluster_df),
                'num_incidencias': len(cluster_df),
                'palabras_clave': self._extract_distinctive_keywords(cluster_df),
                'tipos_principales': self._get_top_ticket_types(cluster_df),
                'ejemplos': self._get_representative_examples(cluster_df)
            }
            
            types[f"tipo_{cluster_id:02d}"] = type_info
            
        return types
    
    def _generate_automatic_type_name(self, cluster_df: pd.DataFrame, cluster_id: int) -> str:
        """Genera nombre COMPLETAMENTE automático basado SOLO en características distintivas"""
        # Extraer todas las palabras de los resúmenes
        all_summaries = ' '.join(cluster_df['Resumen'].fillna('').astype(str))
        
        # Limpiar y tokenizar
        words = re.findall(r'\b[a-zA-Z0-9]{2,}\b', all_summaries.lower())
        
        # Stop words básicas del español
        spanish_stops = {
            'de', 'la', 'el', 'en', 'y', 'a', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 
            'por', 'con', 'su', 'para', 'como', 'las', 'del', 'los', 'un', 'una', 'al', 'del',
            'que', 'fue', 'son', 'han', 'muy', 'más', 'son', 'este', 'esta', 'ese', 'esa'
        }
        
        # Filtrar stop words y palabras muy cortas
        meaningful_words = [w for w in words if w not in spanish_stops and len(w) > 2]
        
        # Contar frecuencias
        word_freq = pd.Series(meaningful_words).value_counts()
        
        # Tomar las 3 palabras más frecuentes como identificadoras únicas
        top_distinctive_words = word_freq.head(3).index.tolist()
        
        if len(top_distinctive_words) >= 2:
            # Crear nombre descriptivo combinando las palabras más distintivas
            primary_word = top_distinctive_words[0].upper()
            secondary_word = top_distinctive_words[1].title()
            
            # Si hay una tercera palabra significativa, incluirla
            if len(top_distinctive_words) >= 3:
                third_word = top_distinctive_words[2].title()
                return f"{primary_word} {secondary_word} {third_word}"
            else:
                return f"{primary_word} {secondary_word}"
        
        elif len(top_distinctive_words) == 1:
            # Solo una palabra distintiva
            return f"Grupo {top_distinctive_words[0].upper()}"
        
        else:
            # Fallback: usar ID del cluster
            return f"Cluster {cluster_id:02d}"
    
    def _extract_distinctive_keywords(self, cluster_df: pd.DataFrame) -> List[str]:
        """Extrae palabras clave MÁS distintivas (no las más comunes)"""
        # Combinar todo el texto del cluster
        cluster_text = ' '.join(cluster_df['Resumen'].fillna('').astype(str))
        
        # Tokenizar
        words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', cluster_text.lower())
        
        # Stop words expandidas
        expanded_stops = {
            'de', 'la', 'el', 'en', 'y', 'a', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 
            'por', 'con', 'su', 'para', 'como', 'las', 'del', 'los', 'un', 'una', 'al', 'del',
            'que', 'fue', 'son', 'han', 'muy', 'más', 'son', 'este', 'esta', 'ese', 'esa',
            'tiene', 'hacer', 'todo', 'año', 'día', 'mes', 'vez', 'caso', 'forma', 'parte'
        }
        
        # Filtrar y contar
        filtered_words = [w for w in words if w not in expanded_stops and len(w) > 2]
        word_freq = pd.Series(filtered_words).value_counts()
        
        # Retornar las 15 más distintivas (no necesariamente las más frecuentes)
        return word_freq.head(15).index.tolist()
    
    def _generate_type_description(self, cluster_df: pd.DataFrame) -> str:
        """Genera descripción para un tipo de incidencia"""
        size = len(cluster_df)
        
        # Obtener las 3 palabras más distintivas para la descripción
        keywords = self._extract_distinctive_keywords(cluster_df)
        main_keywords = keywords[:3] if keywords else ['características', 'específicas']
        
        description = f"Agrupa {size} incidencias que comparten características relacionadas con: {', '.join(main_keywords)}."
        
        return description
    
    def _get_top_ticket_types(self, cluster_df: pd.DataFrame) -> List[str]:
        """Obtiene los principales tipos de ticket"""
        if 'Tipo de ticket' in cluster_df.columns:
            return cluster_df['Tipo de ticket'].value_counts().head(3).index.tolist()
        return []
    
    def _get_representative_examples(self, cluster_df: pd.DataFrame) -> List[Dict]:
        """Obtiene ejemplos representativos"""
        examples = []
        for _, row in cluster_df.head(3).iterrows():
            example = {
                'id': row.get('Ticket ID', f'idx_{row.name}'),
                'resumen': str(row.get('Resumen', ''))[:150] + '...' if len(str(row.get('Resumen', ''))) > 150 else str(row.get('Resumen', '')),
                'tipo_ticket': str(row.get('Tipo de ticket', ''))
            }
            examples.append(example)
        return examples
    
    def _evaluate_system(self, df: pd.DataFrame, cluster_results: Dict, 
                        model_results: Dict) -> Dict[str, float]:
        """Evalúa la performance del sistema"""
        return {
            'silhouette_score': cluster_results.get('silhouette_score', 0.0),
            'model_accuracy': model_results.get('test_accuracy', 0.0),
            'model_cv_score': model_results.get('cv_score', 0.0),
            'num_clusters': len(np.unique(cluster_results['labels'])),
            'coverage': len(df) / len(df)
        }
    
    def _save_model(self):
        """Guarda el modelo entrenado"""
        model_data = {
            'preprocessor': self.preprocessor,
            'clusterer': self.clusterer,
            'classifier': self.classifier,
            'entity_extractor': self.entity_extractor,
            'incident_types': self.incident_types,
            'config': self.config,
            'is_trained': self.is_trained
        }
        
        with open('naturgy_incident_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("💾 Modelo guardado en 'naturgy_incident_model.pkl'")


class TextPreprocessor:
    """Preprocesador de texto con reglas específicas de Naturgy"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_preprocessing_rules()
    
    def setup_preprocessing_rules(self):
        """Configura reglas de preprocesamiento"""
        
        # Stop words seguras (eliminar)
        self.stop_words_safe = [
            'buenos días', 'cordial saludo', 'gracias', 'un saludo', 'saludos',
            'muchas gracias', 'quedo atento', 'quedo atenta', 'favor', 'por favor',
            'adjunto', 'envío', 'enviado', 'estimado', 'estimada', 'hola', 'buen día'
        ]
        
        # Sinonimias para normalización
        self.synonyms = {
            'fallo': 'error', 'incidencia': 'error', 'problema': 'error',
            'rechazo': 'error', 'cancelación': 'baja', 'anulación': 'baja',
            'activación': 'alta', 'creación': 'alta', 'cambiar': 'modificar',
            'actualizar': 'modificar', 'corregir': 'modificar'
        }
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa DataFrame completo"""
        df_processed = df.copy()
        
        # Combinar columnas de texto
        df_processed['combined_text'] = self._combine_text_columns(df_processed)
        
        # Limpiar y normalizar texto
        df_processed['processed_text'] = df_processed['combined_text'].apply(
            self._process_text
        )
        
        return df_processed
    
    def _combine_text_columns(self, df: pd.DataFrame) -> pd.Series:
        """Combina columnas de texto relevantes"""
        text_columns = ['Resumen', 'Notas', 'Tipo de ticket', 'Causa Raíz', 'Resolución']
        combined = []
        
        for _, row in df.iterrows():
            text_parts = []
            for col in text_columns:
                if col in df.columns and pd.notna(row[col]):
                    text_parts.append(str(row[col]))
            combined.append(' '.join(text_parts))
        
        return pd.Series(combined)
    
    def _process_text(self, text: str) -> str:
        """Procesa un texto individual"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower()
        
        # Limpiar caracteres especiales pero mantener espacios y números
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar stop words seguras
        for stop_word in self.stop_words_safe:
            text = text.replace(stop_word.lower(), ' ')
        
        # Aplicar sinonimias
        for synonym, standard in self.synonyms.items():
            text = text.replace(synonym.lower(), standard.lower())
        
        # Limpiar espacios extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class EntityExtractor:
    """Extractor de entidades específicas del dominio"""
    
    def __init__(self):
        self.setup_patterns()
    
    def setup_patterns(self):
        """Configura patrones de extracción"""
        self.patterns = {
            'cups': r'ES\d{4}\d{4}\d{4}\d{4}[A-Z]{2}\d[A-Z]',
            'sr': r'R\d{2}-\d+',
            'req': r'REQ\d+',
            'ofl': r'OFL\d+',
            'fechas': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'productos': r'\b(bono social|cuidahogar|rl1|tur vulnerable)\b',
            'estados': r'\b(activo|inactivo|pendiente|bloqueado|calculable)\b'
        }
    
    def extract_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrae entidades de todos los textos"""
        df_processed = df.copy()
        
        entities_list = []
        for text in df_processed['combined_text']:
            entities = self._extract_from_text(text)
            entities_list.append(entities)
        
        df_processed['entities'] = entities_list
        
        # Crear columnas individuales para cada tipo de entidad
        for entity_type in self.patterns.keys():
            df_processed[f'has_{entity_type}'] = [
                len(entities.get(entity_type, [])) > 0 
                for entities in entities_list
            ]
        
        return df_processed
    
    def _extract_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extrae entidades de un texto"""
        if pd.isna(text):
            return {entity_type: [] for entity_type in self.patterns.keys()}
        
        text = str(text).lower()
        entities = {}
        
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = list(set(matches))
        
        return entities


class IncidentClusterer:
    """Sistema de clustering para agrupar incidencias similares"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.vectorizer = None
        self.clustering_model = None
    
    def cluster_incidents(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Realiza clustering de incidencias"""
        print("🎯 Iniciando clustering de incidencias...")
        
        # Vectorizar textos
        X = self._vectorize_texts(df['processed_text'])
        
        # Determinar número óptimo de clusters
        optimal_k = self._find_optimal_clusters(X)
        
        # Realizar clustering
        labels = self._perform_clustering(X, optimal_k)
        
        # Evaluar clustering
        metrics = self._evaluate_clustering(X, labels)
        
        print(f"✅ Clustering completado: {optimal_k} clusters identificados")
        
        return {
            'labels': labels,
            'n_clusters': optimal_k,
            'vectorizer': self.vectorizer,
            'silhouette_score': metrics.get('silhouette_score', 0.0),
            'cluster_sizes': pd.Series(labels).value_counts().to_dict()
        }
    
    def _vectorize_texts(self, texts: pd.Series) -> np.ndarray:
        """Vectoriza textos usando TF-IDF"""
        self.vectorizer = TfidfVectorizer(
            max_features=self.config['tfidf_max_features'],
            min_df=self.config['tfidf_min_df'],
            max_df=self.config['tfidf_max_df'],
            ngram_range=(1, 2),
            stop_words=None
        )
        
        X = self.vectorizer.fit_transform(texts.fillna(''))
        print(f"📊 Vectorización completada: {X.shape}")
        return X
    
    def _find_optimal_clusters(self, X: np.ndarray) -> int:
        """Encuentra el número óptimo de clusters"""
        max_k = min(self.config['max_clusters'], X.shape[0] // self.config['min_cluster_size'])
        
        if max_k < 2:
            return 2
        
        print(f"🔍 Evaluando clustering desde 2 hasta {max_k} clusters...")
        
        from sklearn.metrics import silhouette_score
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, 30))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.config['random_state'], n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            
            try:
                sil_score = silhouette_score(X, labels)
                silhouette_scores.append(sil_score)
            except:
                silhouette_scores.append(0.0)
        
        # Seleccionar basado en tamaño del dataset para maximizar diversidad
        if X.shape[0] > 3000:
            optimal_k = min(25, max_k)
        elif X.shape[0] > 1000:
            optimal_k = min(15, max_k)
        else:
            optimal_k = min(10, max_k)
        
        print(f"🎯 Número óptimo de clusters seleccionado: {optimal_k}")
        return optimal_k
    
    def _perform_clustering(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Realiza el clustering"""
        self.clustering_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.config['random_state'],
            n_init=10,
            max_iter=300
        )
        
        labels = self.clustering_model.fit_predict(X)
        return labels
    
    def _evaluate_clustering(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evalúa la calidad del clustering"""
        from sklearn.metrics import silhouette_score
        
        try:
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(X, labels)
            else:
                sil_score = 0.0
        except:
            sil_score = 0.0
        
        return {
            'silhouette_score': sil_score,
            'n_clusters': len(np.unique(labels)),
            'largest_cluster_size': np.max(np.bincount(labels))
        }


class PredictiveClassifier:
    """Modelo predictivo para clasificación automática"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.feature_vectorizer = None
        self.label_encoder = None
        self.is_trained = False
    
    def train_model(self, df: pd.DataFrame, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Entrena el modelo predictivo"""
        print("🤖 Entrenando modelo predictivo...")
        
        # Preparar features
        X = self._prepare_features(df)
        y = cluster_labels
        
        # Codificar etiquetas
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=self.config['random_state'],
            stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
        )
        
        # Entrenar modelo
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluar modelo
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        self.is_trained = True
        
        print(f"✅ Modelo entrenado - Accuracy: {test_score:.3f}")
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_score': test_score,
            'model': self.model
        }
    
    def predict(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Predice la clasificación de nuevas incidencias"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        X = self._prepare_features(df)
        
        # Predicción
        prediction = self.model.predict(X)[0]
        
        # Calcular confianza
        try:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
                confidence = np.max(probabilities)
            else:
                confidence = 0.8
        except:
            confidence = 0.5
        
        # Decodificar etiqueta
        original_label = self.label_encoder.inverse_transform([prediction])[0]
        
        return f"tipo_{original_label:02d}", confidence
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepara features para el modelo"""
        if self.feature_vectorizer is None:
            self.feature_vectorizer = TfidfVectorizer(
                max_features=self.config['tfidf_max_features'],
                min_df=self.config['tfidf_min_df'],
                max_df=self.config['tfidf_max_df'],
                ngram_range=(1, 2)
            )
            X_text = self.feature_vectorizer.fit_transform(df['processed_text'].fillna(''))
        else:
            X_text = self.feature_vectorizer.transform(df['processed_text'].fillna(''))
        
        return X_text


# Utilidades para ejecutar el pipeline completo
class IncidentAnalysisRunner:
    """Clase utilitaria para ejecutar análisis completos"""
    
    @staticmethod
    def run_complete_analysis(data_path: str, output_dir: str = '.') -> None:
        """Ejecuta análisis completo y genera reportes"""
        print("🚀 Iniciando análisis completo de incidencias Naturgy...")
        
        # Crear clasificador
        classifier = NaturgyIncidentClassifier()
        
        # Entrenar pipeline
        results = classifier.train_pipeline(data_path)
        
        # Generar reportes
        IncidentAnalysisRunner._generate_reports(results, output_dir)
        
        print("✅ Análisis completo finalizado!")
    
    @staticmethod
    def _generate_reports(results: Dict[str, Any], output_dir: str) -> None:
        """Genera reportes de análisis"""
        
        # Reporte JSON completo
        with open(f"{output_dir}/naturgy_incident_analysis_complete.json", 'w', encoding='utf-8') as f:
            json.dump({
                'tipos_de_incidencia': results['incident_types'],
                'metricas_modelo': results['metrics'],
                'fecha_analisis': datetime.now().isoformat(),
                'cluster_info': {
                    'num_clusters': results['cluster_results']['n_clusters'],
                    'silhouette_score': results['cluster_results']['silhouette_score'],
                    'cluster_sizes': results['cluster_results']['cluster_sizes']
                }
            }, f, ensure_ascii=False, indent=2)
        
        # Reporte de texto legible
        IncidentAnalysisRunner._generate_text_report(results, f"{output_dir}/naturgy_analysis_report.txt")
        
        print(f"📊 Reportes generados en {output_dir}/")
    
    @staticmethod
    def _generate_text_report(results: Dict[str, Any], filepath: str) -> None:
        """Genera reporte de texto legible"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("📊 ANÁLISIS DE INCIDENCIAS NATURGY DELTA - REPORTE COMPLETO\n")
            f.write("=" * 80 + "\n")
            f.write(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total de tipos identificados: {len(results['incident_types'])}\n")
            f.write(f"Accuracy del modelo: {results['metrics'].get('model_accuracy', 0.0):.3f}\n")
            f.write(f"Silhouette Score: {results['metrics'].get('silhouette_score', 0.0):.3f}\n")
            f.write("\n")
            
            # Tipos de incidencia ordenados por volumen
            sorted_types = sorted(
                results['incident_types'].items(),
                key=lambda x: x[1]['num_incidencias'],
                reverse=True
            )
            
            for type_id, type_info in sorted_types:
                f.write(f"🏷️  {type_info['nombre'].upper()}\n")
                f.write("-" * 60 + "\n")
                f.write(f"📊 Incidencias: {type_info['num_incidencias']}\n")
                f.write(f"📝 Descripción: {type_info['descripcion']}\n")
                f.write(f"🔑 Palabras clave: {', '.join(type_info['palabras_clave'][:5])}\n")
                f.write(f"⚡ Tipos principales: {', '.join(type_info['tipos_principales'])}\n")
                f.write("\n")
                
                f.write("📋 Ejemplos representativos:\n")
                for i, example in enumerate(type_info['ejemplos'][:3], 1):
                    f.write(f"  {i}. ID: {example['id']}\n")
                    f.write(f"     Resumen: {example['resumen']}\n")
                    f.write(f"     Tipo: {example['tipo_ticket']}\n")
                    f.write("\n")
                
                f.write("=" * 60 + "\n")
                f.write("\n")


# Script principal para ejecutar desde línea de comandos
if __name__ == "__main__":
    import sys
    
    def main():
        """Función principal para ejecutar el pipeline"""
        print("🚀 NATURGY AI INCIDENT CLASSIFIER - VERSIÓN AUTOMÁTICA")
        print("=" * 60)
        
        if len(sys.argv) < 2:
            print("Uso: python naturgy_ai_incident_classifier_fixed.py <archivo_datos.xlsx> [directorio_salida]")
            print("\nEjemplo:")
            print("  python naturgy_ai_incident_classifier_fixed.py infomation.xlsx ./resultados")
            sys.exit(1)
        
        data_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
        
        try:
            # Ejecutar análisis completo
            IncidentAnalysisRunner.run_complete_analysis(data_path, output_dir)
            
            print("\n" + "=" * 60)
            print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
            print(f"📂 Resultados guardados en: {output_dir}")
            print("📊 Archivos generados:")
            print("  - naturgy_incident_model.pkl (modelo entrenado)")
            print("  - naturgy_incident_analysis_complete.json (análisis completo)")
            print("  - naturgy_analysis_report.txt (reporte legible)")
            
        except Exception as e:
            print(f"❌ Error durante el análisis: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    main()
