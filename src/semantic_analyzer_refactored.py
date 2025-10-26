# Analizador Semántico de Incidencias Técnicas - Versión Refactorizada
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
from typing import List, Dict, Optional
import re
import json
import os
from pathlib import Path
from datetime import datetime

class SemanticOutputManager:
    """Gestiona la estructura de salidas para el analizador semántico"""
    
    def __init__(self, base_output_dir: str = "outputs_semantic"):
        self.base_dir = Path(base_output_dir)
        self.setup_directory_structure()
    
    def setup_directory_structure(self):
        """Crea la estructura de carpetas necesaria"""
        directories = {
            'analysis': self.base_dir / 'analysis',
            'categories': self.base_dir / 'categories',
            'reports': self.base_dir / 'reports',
            'data': self.base_dir / 'data'
        }
        
        for dir_name, dir_path in directories.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Directorio {dir_name} preparado: {dir_path}")
    
    def get_path(self, category: str, filename: str) -> Path:
        """Obtiene la ruta completa para un archivo según su categoría"""
        return self.base_dir / category / filename
    
    def save_categories(self, categories_data: dict, filename: str = 'categorias_semanticas.json'):
        """Guarda las categorías en la carpeta categories"""
        filepath = self.get_path('categories', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(categories_data, f, ensure_ascii=False, indent=2)
        print(f"📊 Categorías guardadas: {filepath}")
        return filepath
    
    def save_analysis(self, analysis_data: dict, filename: str = 'analisis_completo.json'):
        """Guarda el análisis completo en la carpeta analysis"""
        filepath = self.get_path('analysis', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        print(f"🔍 Análisis guardado: {filepath}")
        return filepath
    
    def save_report(self, content: str, filename: str):
        """Guarda reporte de texto en la carpeta reports"""
        filepath = self.get_path('reports', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📄 Reporte guardado: {filepath}")
        return filepath

class SemanticCategoryNaming:
    """Motor de nomenclatura semántica mejorada"""
    
    def __init__(self):
        self.setup_semantic_patterns()
    
    def setup_semantic_patterns(self):
        """Configura patrones semánticos para nomenclatura"""
        
        # Patrones de dominio funcional
        self.functional_patterns = {
            'gestion_datos': {
                'keywords': ['datos', 'data', 'carga', 'actualización', 'masiva', 'bbdd', 'base'],
                'causes': ['actualización masiva de datos', 'carga de datos', 'sincronización'],
                'base_name': 'Gestión de Datos'
            },
            'infraestructura_sistema': {
                'keywords': ['infraestructura', 'sistema', 'servidor', 'red', 'conexión', 'servicio'],
                'causes': ['infraestructura', 'problema de conectividad', 'fallo del sistema'],
                'base_name': 'Infraestructura y Sistemas'
            },
            'comunicacion_notificacion': {
                'keywords': ['comunicación', 'notificación', 'envío', 'correo', 'sms', 'alerta'],
                'causes': ['comunicación', 'notificación', 'envío automático'],
                'base_name': 'Comunicaciones y Notificaciones'
            },
            'procesos_automaticos': {
                'keywords': ['batch', 'proceso', 'job', 'automático', 'programado', 'ejecución'],
                'causes': ['proceso batch', 'job automático', 'ejecución programada'],
                'base_name': 'Procesos Automáticos'
            },
            'consultas_informacion': {
                'keywords': ['consulta', 'información', 'búsqueda', 'listado', 'extracción'],
                'causes': ['consulta funcional', 'solicitud de información', 'extracción de datos'],
                'base_name': 'Consultas e Información'
            },
            'gestion_contratos': {
                'keywords': ['contrato', 'alta', 'baja', 'modificación', 'cambio', 'cliente'],
                'causes': ['gestión de contratos', 'alta de contrato', 'baja de contrato'],
                'base_name': 'Gestión de Contratos'
            },
            'facturacion_cobros': {
                'keywords': ['facturación', 'factura', 'cobro', 'pago', 'importe', 'recibo'],
                'causes': ['facturación', 'cobro', 'gestión económica'],
                'base_name': 'Facturación y Cobros'
            },
            'errores_tecnicos': {
                'keywords': ['error', 'fallo', 'excepción', 'timeout', 'crash', 'bug'],
                'causes': ['error técnico', 'fallo del sistema', 'excepción'],
                'base_name': 'Errores Técnicos'
            }
        }
        
        # Modificadores de contexto
        self.context_modifiers = {
            'urgente': ['urgente', 'crítico', 'inmediato', 'prioritario'],
            'masivo': ['masivo', 'múltiple', 'lote', 'bulk', 'varios'],
            'nocturno': ['nocturno', 'madrugada', 'fuera de horario'],
            'especifico': ['específico', 'puntual', 'individual', 'único']
        }
    
    def generate_semantic_category_name(self, cluster_df: pd.DataFrame, 
                                      top_words: List[str], 
                                      top_causes: List[str]) -> str:
        """Genera nombre semánticamente coherente y técnicamente correcto"""
        
        # Combinar todo el texto para análisis
        all_text = self._combine_cluster_text(cluster_df).lower()
        
        # Identificar patrón funcional principal
        main_pattern = self._identify_functional_pattern(all_text, top_words, top_causes)
        
        # Identificar modificadores de contexto
        context_modifier = self._identify_context_modifier(all_text)
        
        # Generar nombre base
        if main_pattern:
            base_name = self.functional_patterns[main_pattern]['base_name']
        else:
            # Fallback basado en palabras clave
            base_name = self._generate_fallback_name(top_words, top_causes)
        
        # Aplicar modificador si existe
        if context_modifier:
            semantic_name = f"{base_name} {context_modifier.title()}"
        else:
            semantic_name = base_name
        
        # Añadir especificidad basada en el tamaño
        size_spec = self._get_size_specification(len(cluster_df))
        if size_spec:
            semantic_name = f"{semantic_name} - {size_spec}"
        
        return semantic_name
    
    def _combine_cluster_text(self, cluster_df: pd.DataFrame) -> str:
        """Combina todo el texto del cluster para análisis"""
        text_parts = []
        
        for col in ['Resumen', 'Notas', 'Causa Raíz']:
            if col in cluster_df.columns:
                text_parts.extend(cluster_df[col].fillna('').astype(str).tolist())
        
        return ' '.join(text_parts)
    
    def _identify_functional_pattern(self, text: str, words: List[str], 
                                   causes: List[str]) -> Optional[str]:
        """Identifica el patrón funcional principal"""
        pattern_scores = {}
        
        for pattern_name, pattern_info in self.functional_patterns.items():
            score = 0
            
            # Puntuación por keywords
            for keyword in pattern_info['keywords']:
                score += text.count(keyword.lower()) * 2
            
            # Puntuación por palabras clave extraídas
            for word in words:
                if word.lower() in pattern_info['keywords']:
                    score += 3
            
            # Puntuación por causas
            for cause in causes:
                for pattern_cause in pattern_info['causes']:
                    if pattern_cause.lower() in cause.lower():
                        score += 5
            
            pattern_scores[pattern_name] = score
        
        # Retornar el patrón con mayor puntuación si supera el umbral
        if pattern_scores:
            max_pattern = max(pattern_scores.items(), key=lambda x: x[1])
            return max_pattern[0] if max_pattern[1] > 2 else None
        
        return None
    
    def _identify_context_modifier(self, text: str) -> Optional[str]:
        """Identifica modificadores de contexto"""
        for modifier_name, modifier_keywords in self.context_modifiers.items():
            for keyword in modifier_keywords:
                if keyword.lower() in text:
                    return modifier_name
        return None
    
    def _generate_fallback_name(self, words: List[str], causes: List[str]) -> str:
        """Genera nombre alternativo basado en análisis directo"""
        if causes and len(causes) > 0:
            # Usar la causa principal como base
            main_cause = causes[0]
            if len(main_cause) > 30:
                return main_cause[:30] + "..."
            return main_cause
        
        elif words and len(words) > 0:
            # Usar palabras clave más relevantes
            relevant_words = [w for w in words[:3] if len(w) > 3]
            if len(relevant_words) >= 2:
                return f"Incidencias {relevant_words[0].title()} y {relevant_words[1].title()}"
            elif len(relevant_words) == 1:
                return f"Incidencias de {relevant_words[0].title()}"
        
        return "Categoría Técnica Específica"
    
    def _get_size_specification(self, size: int) -> Optional[str]:
        """Obtiene especificación basada en el tamaño del cluster"""
        if size > 100:
            return "Alta Frecuencia"
        elif size > 50:
            return "Frecuencia Media"
        elif size < 10:
            return "Casos Específicos"
        return None

class SemanticIncidentAnalyzer:
    """
    Analiza y categoriza incidencias técnicas con nomenclatura semántica mejorada.
    """
    
    def __init__(self, max_categories=20, output_dir="outputs_semantic"):
        self.max_categories = max_categories
        self.output_manager = SemanticOutputManager(output_dir)
        self.naming_engine = SemanticCategoryNaming()
        
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words=None
        )
        self.categories = {}
        self.incident_vectors = None
        
    def clean_text(self, text):
        """Limpia y normaliza el texto"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower()
        # Mantener caracteres españoles y técnicos importantes
        text = re.sub(r'[^a-záéíóúñü0-9\s\-_]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_technical_keywords(self, text):
        """Extrae palabras clave técnicas específicas"""
        keywords = []
        
        # Patrones técnicos comunes
        patterns = {
            'error_types': r'\b(fail|error|exception|timeout|crash|bug)\b',
            'systems': r'\b(delta|atlas|ppm|oracle|mysql|database|sql)\b',
            'operations': r'\b(batch|job|process|script|execution|run)\b',
            'network': r'\b(connection|network|ftp|http|api|service)\b',
            'data': r'\b(datos|data|carga|load|backup|restore|sync)\b',
            'infrastructure': r'\b(server|infraestructura|sistema|aplicacion|app)\b'
        }
        
        text_lower = text.lower()
        for category, pattern in patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                keywords.extend([f"{category}_{match}" for match in matches])
        
        return ' '.join(keywords)
    
    def prepare_text_for_analysis(self, df):
        """Prepara los textos combinados para análisis"""
        combined_texts = []
        
        for idx, row in df.iterrows():
            # Combinar todos los campos de texto relevantes
            text_parts = []
            
            if not pd.isna(row.get('Resumen', '')):
                text_parts.append(self.clean_text(row['Resumen']))
            
            if not pd.isna(row.get('Notas', '')):
                text_parts.append(self.clean_text(row['Notas']))
            
            if not pd.isna(row.get('Causa Raíz', '')):
                text_parts.append(self.clean_text(row['Causa Raíz']))
            
            # Extraer keywords técnicos
            full_text = ' '.join(text_parts)
            technical_keywords = self.extract_technical_keywords(full_text)
            
            # Combinar texto limpio con keywords técnicos
            final_text = f"{full_text} {technical_keywords}"
            combined_texts.append(final_text)
        
        return combined_texts
    
    def analyze_incidents(self, df):
        """Analiza las incidencias y las agrupa en categorías semánticas mejoradas"""
        print("🔍 Iniciando análisis semántico avanzado de incidencias...")
        
        # Preparar textos
        texts = self.prepare_text_for_analysis(df)
        
        # Vectorizar textos
        try:
            X = self.vectorizer.fit_transform(texts)
            self.incident_vectors = X
            print(f"✅ Vectorización completada: {X.shape}")
        except Exception as e:
            print(f"❌ Error en vectorización: {e}")
            return None
        
        # Clustering inicial con más clusters para luego fusionar
        initial_clusters = min(30, len(df) // 20)  # Empezar con más clusters
        print(f"🎯 Clustering inicial con {initial_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=initial_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Analizar clusters y crear categorías preliminares
        preliminary_categories = self._analyze_clusters_improved(df, texts, cluster_labels, kmeans)
        
        # Fusionar categorías similares hasta llegar a máximo permitido
        final_categories = self._merge_similar_categories(preliminary_categories, df)
        
        # Asignar nombres y descripciones finales con nomenclatura mejorada
        self.categories = self._finalize_categories_improved(final_categories, df)
        
        print(f"✅ Análisis completado: {len(self.categories)} categorías identificadas")
        
        # Guardar resultados organizados
        self._save_organized_results(df)
        
        return self.categories
    
    def _analyze_clusters_improved(self, df, texts, cluster_labels, kmeans):
        """Analiza cada cluster con técnicas mejoradas"""
        categories = {}
        
        for cluster_id in range(len(set(cluster_labels))):
            # Obtener índices del cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Obtener textos del cluster
            cluster_texts = [texts[i] for i in cluster_indices]
            cluster_df = df.iloc[cluster_indices].copy()
            
            # Analizar palabras más frecuentes con filtros mejorados
            cluster_text = ' '.join(cluster_texts)
            words = self._extract_meaningful_words(cluster_text)
            
            # Analizar causas raíz más comunes
            if 'Causa Raíz' in cluster_df.columns:
                cause_counts = cluster_df['Causa Raíz'].value_counts()
                top_causes = cause_counts.head(3).index.tolist()
            else:
                top_causes = []
            
            # Crear categoría preliminar con información mejorada
            categories[cluster_id] = {
                'incidents': cluster_indices.tolist(),
                'top_words': words[:8],
                'top_causes': top_causes,
                'size': len(cluster_indices),
                'cluster_center': kmeans.cluster_centers_[cluster_id],
                'semantic_features': self._extract_semantic_features(cluster_df)
            }
        
        return categories
    
    def _extract_meaningful_words(self, text: str) -> List[str]:
        """Extrae palabras significativas con filtros avanzados"""
        words = text.split()
        
        # Stop words técnicas expandidas
        stop_words = {
            'de', 'la', 'el', 'en', 'y', 'a', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su',
            'por', 'con', 'para', 'como', 'las', 'del', 'los', 'un', 'una', 'al',
            'que', 'fue', 'son', 'han', 'muy', 'más', 'este', 'esta', 'ese', 'esa',
            'tiene', 'hacer', 'todo', 'año', 'día', 'mes', 'vez', 'caso', 'forma', 'parte',
            'pero', 'sin', 'sobre', 'entre', 'hasta', 'donde', 'cuando', 'porque'
        }
        
        # Filtrar palabras significativas
        meaningful_words = []
        for word in words:
            if (len(word) > 3 and 
                word.lower() not in stop_words and
                not word.isdigit() and
                any(c.isalpha() for c in word)):
                meaningful_words.append(word.lower())
        
        # Contar frecuencias y retornar las más importantes
        word_freq = Counter(meaningful_words)
        return [word for word, freq in word_freq.most_common(10) if freq > 1]
    
    def _extract_semantic_features(self, cluster_df: pd.DataFrame) -> Dict:
        """Extrae características semánticas del cluster"""
        features = {
            'avg_text_length': 0,
            'has_technical_terms': False,
            'urgency_indicators': 0,
            'time_patterns': []
        }
        
        if 'Resumen' in cluster_df.columns:
            # Longitud promedio de texto
            text_lengths = cluster_df['Resumen'].fillna('').astype(str).apply(len)
            features['avg_text_length'] = text_lengths.mean()
            
            # Términos técnicos
            all_text = ' '.join(cluster_df['Resumen'].fillna('').astype(str)).lower()
            technical_terms = ['sistema', 'error', 'fallo', 'proceso', 'batch', 'datos']
            features['has_technical_terms'] = any(term in all_text for term in technical_terms)
            
            # Indicadores de urgencia
            urgency_terms = ['urgente', 'crítico', 'inmediato', 'bloqueo']
            features['urgency_indicators'] = sum(all_text.count(term) for term in urgency_terms)
        
        return features
    
    def _merge_similar_categories(self, categories, df):
        """Fusiona categorías similares con criterios mejorados"""
        print(f"🔄 Fusionando categorías similares (inicial: {len(categories)}, objetivo: ≤{self.max_categories})")
        
        while len(categories) > self.max_categories:
            # Encontrar las dos categorías más similares
            max_similarity = -1
            merge_pair = None
            
            cat_ids = list(categories.keys())
            for i in range(len(cat_ids)):
                for j in range(i + 1, len(cat_ids)):
                    cat1, cat2 = cat_ids[i], cat_ids[j]
                    
                    # Calcular similitud mejorada
                    sim_score = self._calculate_enhanced_similarity(
                        categories[cat1], categories[cat2]
                    )
                    
                    if sim_score > max_similarity:
                        max_similarity = sim_score
                        merge_pair = (cat1, cat2)
            
            # Fusionar las categorías más similares
            if merge_pair and max_similarity > 0.1:  # Umbral mínimo
                cat1, cat2 = merge_pair
                merged_category = self._merge_two_categories_improved(
                    categories[cat1], categories[cat2]
                )
                
                # Reemplazar con la categoría fusionada
                new_id = min(cat1, cat2)
                categories[new_id] = merged_category
                del categories[max(cat1, cat2)]
                
                print(f"  Fusionadas categorías {cat1} y {cat2} (similitud: {max_similarity:.3f})")
            else:
                # Si no hay suficiente similitud, parar la fusión
                break
        
        return categories
    
    def _calculate_enhanced_similarity(self, cat1, cat2):
        """Calcula similitud mejorada entre categorías"""
        # Similitud de palabras clave (40%)
        words1 = set(cat1['top_words'])
        words2 = set(cat2['top_words'])
        word_sim = len(words1.intersection(words2)) / len(words1.union(words2)) if words1.union(words2) else 0
        
        # Similitud de causas raíz (40%)
        causes1 = set(cat1['top_causes'])
        causes2 = set(cat2['top_causes'])
        cause_sim = len(causes1.intersection(causes2)) / len(causes1.union(causes2)) if causes1.union(causes2) else 0
        
        # Similitud de características semánticas (20%)
        semantic_sim = 0
        if 'semantic_features' in cat1 and 'semantic_features' in cat2:
            feat1 = cat1['semantic_features']
            feat2 = cat2['semantic_features']
            
            # Comparar características booleanas
            if feat1['has_technical_terms'] == feat2['has_technical_terms']:
                semantic_sim += 0.5
            
            # Comparar indicadores de urgencia
            if abs(feat1['urgency_indicators'] - feat2['urgency_indicators']) <= 1:
                semantic_sim += 0.5
        
        # Combinar similitudes con pesos
        total_sim = 0.4 * word_sim + 0.4 * cause_sim + 0.2 * semantic_sim
        return total_sim
    
    def _merge_two_categories_improved(self, cat1, cat2):
        """Fusiona dos categorías con información mejorada"""
        merged = {
            'incidents': cat1['incidents'] + cat2['incidents'],
            'size': cat1['size'] + cat2['size']
        }
        
        # Combinar palabras clave (mantener las más frecuentes)
        all_words = cat1['top_words'] + cat2['top_words']
        word_counts = Counter(all_words)
        merged['top_words'] = [word for word, count in word_counts.most_common(8)]
        
        # Combinar causas raíz
        all_causes = cat1['top_causes'] + cat2['top_causes']
        cause_counts = Counter(all_causes)
        merged['top_causes'] = [cause for cause, count in cause_counts.most_common(3)]
        
        # Promedio de centros de cluster
        if 'cluster_center' in cat1 and 'cluster_center' in cat2:
            merged['cluster_center'] = (cat1['cluster_center'] + cat2['cluster_center']) / 2
        
        # Combinar características semánticas
        if 'semantic_features' in cat1 and 'semantic_features' in cat2:
            merged['semantic_features'] = {
                'avg_text_length': (cat1['semantic_features']['avg_text_length'] + 
                                  cat2['semantic_features']['avg_text_length']) / 2,
                'has_technical_terms': (cat1['semantic_features']['has_technical_terms'] or 
                                      cat2['semantic_features']['has_technical_terms']),
                'urgency_indicators': (cat1['semantic_features']['urgency_indicators'] + 
                                     cat2['semantic_features']['urgency_indicators'])
            }
        
        return merged
    
    def _finalize_categories_improved(self, categories, df):
        """Asigna nombres y descripciones finales con nomenclatura semántica mejorada"""
        final_categories = {}
        
        for i, (cat_id, category) in enumerate(categories.items()):
            # Generar nombre semánticamente coherente
            semantic_name = self.naming_engine.generate_semantic_category_name(
                df.iloc[category['incidents']], 
                category['top_words'], 
                category['top_causes']
            )
            
            # Generar descripción técnica detallada
            description = self._generate_enhanced_description(category, df, semantic_name)
            
            # Obtener ejemplos representativos
            examples = self._get_enhanced_examples(category, df)
            
            # Evaluar criticidad y prioridad
            criticality = self._assess_category_criticality(category, df)
            
            # Crear clave semánticamente significativa
            safe_key = re.sub(r'[^\w\s-]', '', semantic_name.lower())
            safe_key = re.sub(r'[\s-]+', '_', safe_key)
            
            final_categories[safe_key] = {
                'nombre': semantic_name,
                'descripcion': description,
                'ejemplos': examples,
                'num_incidencias': category['size'],
                'incident_ids': category['incidents'][:10],
                'palabras_clave': category['top_words'],
                'causas_principales': category['top_causes'],
                'criticidad': criticality,
                'caracteristicas_tecnicas': category.get('semantic_features', {}),
                'porcentaje_total': round((category['size'] / len(df)) * 100, 2)
            }
        
        return final_categories
    
    def _generate_enhanced_description(self, category, df, semantic_name):
        """Genera descripción técnica mejorada"""
        size = category['size']
        causes = category['top_causes']
        words = category['top_words']
        
        description = f"Categoría '{semantic_name}' que engloba {size} incidencias técnicas "
        
        # Descripción basada en características semánticas
        if 'semantic_features' in category:
            features = category['semantic_features']
            if features['has_technical_terms']:
                description += "de naturaleza técnica especializada, "
            if features['urgency_indicators'] > 0:
                description += "que requieren atención prioritaria, "
        
        # Contexto funcional
        if causes:
            main_cause = causes[0]
            description += f"principalmente relacionadas con '{main_cause}'. "
            
            if len(causes) > 1:
                description += f"Causas secundarias incluyen: {', '.join(causes[1:2])}. "
        
        # Patrones técnicos identificados
        tech_words = [w for w in words if len(w) > 4]
        if tech_words:
            description += f"Términos técnicos frecuentes: {', '.join(tech_words[:4])}."
        
        return description
    
    def _get_enhanced_examples(self, category, df):
        """Obtiene ejemplos representativos mejorados"""
        incidents = category['incidents'][:5]
        examples = []
        
        for idx in incidents:
            if idx < len(df):
                row = df.iloc[idx]
                example = {
                    'id': row.get('Ticket ID', f'idx_{idx}'),
                    'resumen': str(row.get('Resumen', ''))[:120] + '...' if len(str(row.get('Resumen', ''))) > 120 else str(row.get('Resumen', '')),
                    'causa_raiz': str(row.get('Causa Raíz', '')),
                    'prioridad': row.get('Prioridad', 'No especificada')
                }
                examples.append(example)
        
        return examples
    
    def _assess_category_criticality(self, category, df):
        """Evalúa la criticidad de la categoría"""
        if 'semantic_features' not in category:
            return 'Media'
        
        features = category['semantic_features']
        urgency_score = features.get('urgency_indicators', 0)
        
        if urgency_score > 5:
            return 'Alta'
        elif urgency_score > 2:
            return 'Media'
        else:
            return 'Baja'
    
    def _save_organized_results(self, df):
        """Guarda los resultados con estructura organizada"""
        
        # 1. Guardar categorías con metadatos completos
        categories_data = {
            'metadata': {
                'fecha_analisis': datetime.now().isoformat(),
                'total_incidencias_analizadas': len(df),
                'total_categorias': len(self.categories),
                'metodo_analisis': 'Análisis Semántico Avanzado',
                'max_categorias_permitidas': self.max_categories
            },
            'categorias': self.categories
        }
        
        self.output_manager.save_categories(categories_data)
        
        # 2. Guardar análisis detallado
        analysis_data = {
            'resumen_ejecutivo': self._generate_executive_summary(df),
            'estadisticas_clustering': self._get_clustering_stats(df),
            'distribuciones': self._get_category_distributions(),
            'recomendaciones': self._generate_recommendations()
        }
        
        self.output_manager.save_analysis(analysis_data)
        
        # 3. Generar y guardar reportes
        comprehensive_report = self.generate_comprehensive_report()
        self.output_manager.save_report(comprehensive_report, 'reporte_semantico_completo.txt')
        
        technical_summary = self._generate_technical_summary()
        self.output_manager.save_report(technical_summary, 'resumen_tecnico.txt')
    
    def _generate_executive_summary(self, df):
        """Genera resumen ejecutivo del análisis"""
        total_categories = len(self.categories)
        high_criticality = sum(1 for cat in self.categories.values() if cat.get('criticidad') == 'Alta')
        
        return {
            'total_incidencias': len(df),
            'categorias_identificadas': total_categories,
            'categorias_alta_criticidad': high_criticality,
            'cobertura_analisis': f"{(total_categories / self.max_categories) * 100:.1f}%",
            'categoria_mas_frecuente': max(self.categories.items(), key=lambda x: x[1]['num_incidencias'])[1]['nombre'] if self.categories else 'N/A'
        }
    
    def _get_clustering_stats(self, df):
        """Obtiene estadísticas del clustering"""
        if not self.categories:
            return {}
        
        sizes = [cat['num_incidencias'] for cat in self.categories.values()]
        return {
            'tamano_promedio_categoria': np.mean(sizes),
            'tamano_mediano_categoria': np.median(sizes),
            'categoria_mas_grande': max(sizes),
            'categoria_mas_pequena': min(sizes),
            'desviacion_estandar': np.std(sizes)
        }
    
    def _get_category_distributions(self):
        """Obtiene distribuciones de las categorías"""
        if not self.categories:
            return {}
        
        # Distribución por criticidad
        criticality_dist = {}
        for cat in self.categories.values():
            crit = cat.get('criticidad', 'No evaluada')
            criticality_dist[crit] = criticality_dist.get(crit, 0) + 1
        
        # Distribución por tamaño
        size_dist = {'Pequeñas (<20)': 0, 'Medianas (20-50)': 0, 'Grandes (>50)': 0}
        for cat in self.categories.values():
            size = cat['num_incidencias']
            if size < 20:
                size_dist['Pequeñas (<20)'] += 1
            elif size <= 50:
                size_dist['Medianas (20-50)'] += 1
            else:
                size_dist['Grandes (>50)'] += 1
        
        return {
            'por_criticidad': criticality_dist,
            'por_tamano': size_dist
        }
    
    def _generate_recommendations(self):
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []
        
        if not self.categories:
            return ['No se pudieron generar recomendaciones - No hay categorías disponibles']
        
        # Recomendaciones basadas en criticidad
        high_crit_cats = [cat for cat in self.categories.values() if cat.get('criticidad') == 'Alta']
        if high_crit_cats:
            recommendations.append(f"Priorizar atención a {len(high_crit_cats)} categorías de alta criticidad")
        
        # Recomendaciones basadas en volumen
        large_cats = [cat for cat in self.categories.values() if cat['num_incidencias'] > 50]
        if large_cats:
            recommendations.append(f"Implementar automatización en {len(large_cats)} categorías de alto volumen")
        
        # Recomendaciones generales
        recommendations.extend([
            "Desarrollar procedimientos específicos por categoría identificada",
            "Establecer KPIs de seguimiento por tipo de incidencia",
            "Revisar y actualizar el análisis periódicamente"
        ])
        
        return recommendations
    
    def generate_comprehensive_report(self):
        """Genera reporte completo del análisis semántico"""
        if not self.categories:
            return "No hay categorías disponibles para generar el reporte."
        
        report = []
        report.append("=" * 80)
        report.append("📊 ANÁLISIS SEMÁNTICO AVANZADO DE INCIDENCIAS TÉCNICAS")
        report.append("=" * 80)
        report.append(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total de categorías identificadas: {len(self.categories)}")
        report.append(f"Máximo permitido: {self.max_categories}")
        report.append("")
        
        # Ordenar categorías por número de incidencias (descendente)
        sorted_categories = sorted(
            self.categories.items(), 
            key=lambda x: x[1]['num_incidencias'], 
            reverse=True
        )
        
        report.append("🏷️  CATEGORÍAS SEMÁNTICAMENTE IDENTIFICADAS")
        report.append("=" * 80)
        
        for cat_key, category in sorted_categories:
            report.append(f"\n📋 {category['nombre'].upper()}")
            report.append("-" * 60)
            report.append(f"📊 Incidencias: {category['num_incidencias']} ({category.get('porcentaje_total', 0)}% del total)")
            report.append(f"⚡ Criticidad: {category.get('criticidad', 'No evaluada')}")
            report.append(f"📝 Descripción: {category['descripcion']}")
            report.append(f"🔑 Palabras clave: {', '.join(category['palabras_clave'])}")
            report.append(f"⚡ Causas principales: {', '.join(category['causas_principales'])}")
            
            # Características técnicas
            if 'caracteristicas_tecnicas' in category:
                features = category['caracteristicas_tecnicas']
                if features.get('has_technical_terms'):
                    report.append("🔧 Naturaleza: Técnica especializada")
                if features.get('urgency_indicators', 0) > 0:
                    report.append(f"⚠️  Indicadores de urgencia: {features['urgency_indicators']}")
            
            report.append("")
            report.append("📋 Ejemplos representativos:")
            
            for i, example in enumerate(category['ejemplos'][:3], 1):
                report.append(f"  {i}. ID: {example['id']}")
                report.append(f"     Resumen: {example['resumen']}")
                report.append(f"     Causa: {example['causa_raiz']}")
                if 'prioridad' in example:
                    report.append(f"     Prioridad: {example['prioridad']}")
                report.append("")
            
            report.append("=" * 60)
            report.append("")
        
        return '\n'.join(report)
    
    def _generate_technical_summary(self):
        """Genera resumen técnico del análisis"""
        summary = []
        summary.append("=" * 60)
        summary.append("🔧 RESUMEN TÉCNICO - ANÁLISIS SEMÁNTICO")
        summary.append("=" * 60)
        summary.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d')}")
        summary.append("")
        
        if not self.categories:
            summary.append("❌ No se generaron categorías en el análisis")
            return '\n'.join(summary)
        
        # Estadísticas técnicas
        sizes = [cat['num_incidencias'] for cat in self.categories.values()]
        summary.append("📊 ESTADÍSTICAS TÉCNICAS:")
        summary.append(f"• Total de categorías: {len(self.categories)}")
        summary.append(f"• Promedio incidencias por categoría: {np.mean(sizes):.1f}")
        summary.append(f"• Categoría más grande: {max(sizes)} incidencias")
        summary.append(f"• Categoría más pequeña: {min(sizes)} incidencias")
        summary.append("")
        
        # Top 5 por volumen
        sorted_cats = sorted(self.categories.items(), key=lambda x: x[1]['num_incidencias'], reverse=True)[:5]
        summary.append("🎯 TOP 5 CATEGORÍAS POR VOLUMEN:")
        for i, (key, cat) in enumerate(sorted_cats, 1):
            summary.append(f"{i}. {cat['nombre']} - {cat['num_incidencias']} incidencias")
        
        summary.append("")
        summary.append("⚡ DISTRIBUCIÓN POR CRITICIDAD:")
        crit_dist = {}
        for cat in self.categories.values():
            crit = cat.get('criticidad', 'No evaluada')
            crit_dist[crit] = crit_dist.get(crit, 0) + 1
        
        for crit, count in crit_dist.items():
            summary.append(f"• {crit}: {count} categorías")
        
        return '\n'.join(summary)
    
    def export_to_json(self, filename='categorias_semanticas.json'):
        """Exporta las categorías a JSON (método de compatibilidad)"""
        if not self.categories:
            print("❌ No hay categorías para exportar")
            return
        
        # Usar el output manager para guardar
        categories_data = {
            'metadata': {
                'fecha_analisis': datetime.now().isoformat(),
                'total_categorias': len(self.categories),
                'max_categorias_permitidas': self.max_categories,
                'metodo': 'Análisis Semántico Refactorizado'
            },
            'categorias': self.categories
        }
        
        self.output_manager.save_categories(categories_data, filename)

def main():
    """Función principal para ejecutar el análisis semántico refactorizado"""
    print("🚀 Iniciando Analizador Semántico Refactorizado")
    
    # Cargar datos
    try:
        df = pd.read_excel('infomation.xlsx')
        print(f"✅ Datos cargados: {df.shape[0]} registros")
    except Exception as e:
        try:
            df = pd.read_csv('infomation.csv', encoding='ISO-8859-1', sep=';')
            print(f"✅ Datos cargados desde CSV: {df.shape[0]} registros")
        except Exception as e2:
            print(f"❌ Error cargando datos: {e}")
            print(f"❌ Error alternativo CSV: {e2}")
            return
    
    # Filtrar registros válidos
    df_clean = df[['Ticket ID', 'Resumen', 'Notas', 'Causa Raíz']].copy()
    df_clean = df_clean.dropna(subset=['Causa Raíz'])
    print(f"✅ Registros válidos: {df_clean.shape[0]}")
    
    # Crear analizador con salida organizada
    analyzer = SemanticIncidentAnalyzer(max_categories=20, output_dir="outputs_semantic")
    
    # Ejecutar análisis
    categories = analyzer.analyze_incidents(df_clean)
    
    if categories:
        print("\n" + "=" * 65)
        print("✅ ANÁLISIS SEMÁNTICO COMPLETADO")
        print("📊 Mejoras implementadas:")
        print("  ✓ Estructura de carpetas organizada")
        print("  ✓ Nomenclatura semánticamente coherente")
        print("  ✓ Análisis de criticidad incluido")
        print("  ✓ Reportes categorizados por tipo")
        print("  ✓ Características técnicas evaluadas")
    else:
        print("❌ No se pudieron generar categorías")

if __name__ == "__main__":
    main()
