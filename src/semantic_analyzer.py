# Analizador Semántico de Incidencias Técnicas
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import re
import json
from datetime import datetime

class SemanticIncidentAnalyzer:
    """
    Analiza y categoriza incidencias técnicas en máximo 20 tipos semánticos.
    """
    
    def __init__(self, max_categories=20):
        self.max_categories = max_categories
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
            
            if not pd.isna(row.get('Causa Raiz', '')):
                text_parts.append(self.clean_text(row['Causa Raiz']))
            
            # Extraer keywords técnicos
            full_text = ' '.join(text_parts)
            technical_keywords = self.extract_technical_keywords(full_text)
            
            # Combinar texto limpio con keywords técnicos
            final_text = f"{full_text} {technical_keywords}"
            combined_texts.append(final_text)
        
        return combined_texts
    
    def analyze_incidents(self, df):
        """Analiza las incidencias y las agrupa en categorías semánticas"""
        print("🔍 Iniciando análisis semántico de incidencias...")
        
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
        preliminary_categories = self._analyze_clusters(df, texts, cluster_labels, kmeans)
        
        # Fusionar categorías similares hasta llegar a máximo 20
        final_categories = self._merge_similar_categories(preliminary_categories, df)
        
        # Asignar nombres y descripciones finales
        self.categories = self._finalize_categories(final_categories, df)
        
        print(f"✅ Análisis completado: {len(self.categories)} categorías identificadas")
        return self.categories
    
    def _analyze_clusters(self, df, texts, cluster_labels, kmeans):
        """Analiza cada cluster para identificar patrones"""
        categories = {}
        
        for cluster_id in range(len(set(cluster_labels))):
            # Obtener índices del cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Obtener textos del cluster
            cluster_texts = [texts[i] for i in cluster_indices]
            cluster_df = df.iloc[cluster_indices].copy()
            
            # Analizar palabras más frecuentes
            cluster_text = ' '.join(cluster_texts)
            words = cluster_text.split()
            word_freq = Counter(words)
            top_words = [word for word, freq in word_freq.most_common(10) if len(word) > 2]
            
            # Analizar causas raíz más comunes
            cause_counts = cluster_df['Causa Raiz'].value_counts()
            top_causes = cause_counts.head(3).index.tolist()
            
            # Crear categoría preliminar
            categories[cluster_id] = {
                'incidents': cluster_indices.tolist(),
                'top_words': top_words[:5],
                'top_causes': top_causes,
                'size': len(cluster_indices),
                'cluster_center': kmeans.cluster_centers_[cluster_id]
            }
        
        return categories
    
    def _merge_similar_categories(self, categories, df):
        """Fusiona categorías similares hasta llegar al máximo permitido"""
        print(f"🔄 Fusionando categorías similares (inicial: {len(categories)}, objetivo: ≤{self.max_categories})")
        
        while len(categories) > self.max_categories:
            # Encontrar las dos categorías más similares
            max_similarity = -1
            merge_pair = None
            
            cat_ids = list(categories.keys())
            for i in range(len(cat_ids)):
                for j in range(i + 1, len(cat_ids)):
                    cat1, cat2 = cat_ids[i], cat_ids[j]
                    
                    # Calcular similitud basada en palabras clave y causas
                    sim_score = self._calculate_category_similarity(
                        categories[cat1], categories[cat2]
                    )
                    
                    if sim_score > max_similarity:
                        max_similarity = sim_score
                        merge_pair = (cat1, cat2)
            
            # Fusionar las categorías más similares
            if merge_pair:
                cat1, cat2 = merge_pair
                merged_category = self._merge_two_categories(
                    categories[cat1], categories[cat2]
                )
                
                # Reemplazar con la categoría fusionada
                new_id = min(cat1, cat2)
                categories[new_id] = merged_category
                del categories[max(cat1, cat2)]
                
                print(f"  Fusionadas categorías {cat1} y {cat2} (similitud: {max_similarity:.3f})")
        
        return categories
    
    def _calculate_category_similarity(self, cat1, cat2):
        """Calcula similitud entre dos categorías"""
        # Similitud de palabras clave
        words1 = set(cat1['top_words'])
        words2 = set(cat2['top_words'])
        word_sim = len(words1.intersection(words2)) / len(words1.union(words2)) if words1.union(words2) else 0
        
        # Similitud de causas raíz
        causes1 = set(cat1['top_causes'])
        causes2 = set(cat2['top_causes'])
        cause_sim = len(causes1.intersection(causes2)) / len(causes1.union(causes2)) if causes1.union(causes2) else 0
        
        # Similitud de vectores (si están disponibles)
        if 'cluster_center' in cat1 and 'cluster_center' in cat2:
            center_sim = cosine_similarity([cat1['cluster_center']], [cat2['cluster_center']])[0][0]
        else:
            center_sim = 0
        
        # Combinar similitudes
        total_sim = 0.4 * word_sim + 0.4 * cause_sim + 0.2 * center_sim
        return total_sim
    
    def _merge_two_categories(self, cat1, cat2):
        """Fusiona dos categorías en una"""
        merged = {
            'incidents': cat1['incidents'] + cat2['incidents'],
            'size': cat1['size'] + cat2['size']
        }
        
        # Combinar palabras clave (tomar las más frecuentes)
        all_words = cat1['top_words'] + cat2['top_words']
        word_counts = Counter(all_words)
        merged['top_words'] = [word for word, count in word_counts.most_common(5)]
        
        # Combinar causas raíz (tomar las más comunes)
        all_causes = cat1['top_causes'] + cat2['top_causes']
        cause_counts = Counter(all_causes)
        merged['top_causes'] = [cause for cause, count in cause_counts.most_common(3)]
        
        # Promedio de centros de cluster
        if 'cluster_center' in cat1 and 'cluster_center' in cat2:
            merged['cluster_center'] = (cat1['cluster_center'] + cat2['cluster_center']) / 2
        
        return merged
    
    def _finalize_categories(self, categories, df):
        """Asigna nombres y descripciones finales a las categorías"""
        final_categories = {}
        
        for i, (cat_id, category) in enumerate(categories.items()):
            # Generar nombre basado en palabras clave y causas más comunes
            name = self._generate_category_name(category)
            
            # Generar descripción
            description = self._generate_category_description(category, df)
            
            # Obtener ejemplos representativos
            examples = self._get_representative_examples(category, df)
            
            final_categories[f"tipo_{i+1:02d}"] = {
                'nombre': name,
                'descripcion': description,
                'ejemplos': examples,
                'num_incidencias': category['size'],
                'incident_ids': category['incidents'][:10],  # Máximo 10 IDs
                'palabras_clave': category['top_words'],
                'causas_principales': category['top_causes']
            }
        
        return final_categories
    
    def _generate_category_name(self, category):
        """Genera nombre para la categoría basado en patrones"""
        words = category['top_words']
        causes = category['top_causes']
        
        # Patrones para generar nombres técnicos
        if any('error' in word or 'fail' in word for word in words):
            if any('infraestructura' in cause.lower() for cause in causes):
                return "Errores de Infraestructura"
            elif any('comunicacion' in cause.lower() for cause in causes):
                return "Errores de Comunicación"
            else:
                return "Errores de Sistema"
        
        elif any('datos' in word or 'data' in word for word in words):
            if any('masiva' in cause.lower() for cause in causes):
                return "Actualización Masiva Datos"
            else:
                return "Gestión de Datos"
        
        elif any('consulta' in cause.lower() for cause in causes):
            return "Consultas Funcionales"
        
        elif any('batch' in word or 'job' in word for word in words):
            return "Procesos Batch"
        
        elif any('extraccion' in cause.lower() or 'listado' in cause.lower() for cause in causes):
            return "Extracciones y Listados"
        
        elif any('ticket' in cause.lower() and 'gestionable' in cause.lower() for cause in causes):
            return "Tickets No Gestionables"
        
        else:
            # Usar la causa más común como base del nombre
            if causes:
                return causes[0][:30] + "..." if len(causes[0]) > 30 else causes[0]
            else:
                return f"Categoría Técnica"
    
    def _generate_category_description(self, category, df):
        """Genera descripción detallada de la categoría"""
        size = category['size']
        causes = category['top_causes']
        words = category['top_words']
        
        description = f"Agrupa {size} incidencias relacionadas con "
        
        if causes:
            main_cause = causes[0].lower()
            if 'error' in main_cause:
                description += "errores técnicos y fallos del sistema. "
            elif 'datos' in main_cause:
                description += "gestión y actualización de datos. "
            elif 'consulta' in main_cause:
                description += "consultas funcionales y soporte al usuario. "
            elif 'infraestructura' in main_cause:
                description += "problemas de infraestructura y servicios. "
            else:
                description += f"'{causes[0]}'. "
        
        if len(causes) > 1:
            description += f"Las causas principales incluyen: {', '.join(causes[:2])}."
        
        # Añadir información sobre palabras clave técnicas
        tech_words = [w for w in words if any(t in w for t in ['error', 'batch', 'data', 'job', 'fail'])]
        if tech_words:
            description += f" Términos técnicos frecuentes: {', '.join(tech_words[:3])}."
        
        return description
    
    def _get_representative_examples(self, category, df):
        """Obtiene ejemplos representativos de la categoría"""
        incidents = category['incidents'][:5]  # Máximo 5 ejemplos
        examples = []
        
        for idx in incidents:
            row = df.iloc[idx]
            example = {
                'id': row.get('Ticket ID', f'idx_{idx}'),
                'resumen': str(row.get('Resumen', ''))[:100] + '...' if len(str(row.get('Resumen', ''))) > 100 else str(row.get('Resumen', '')),
                'causa_raiz': str(row.get('Causa Raiz', ''))
            }
            examples.append(example)
        
        return examples
    
    def generate_report(self):
        """Genera reporte completo del análisis"""
        if not self.categories:
            return "No hay categorías disponibles. Ejecuta analyze_incidents() primero."
        
        report = []
        report.append("=" * 80)
        report.append("📊 ANÁLISIS SEMÁNTICO DE INCIDENCIAS TÉCNICAS")
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
        
        for cat_id, category in sorted_categories:
            report.append(f"🏷️  {category['nombre'].upper()}")
            report.append("-" * 60)
            report.append(f"📊 Incidencias: {category['num_incidencias']}")
            report.append(f"📝 Descripción: {category['descripcion']}")
            report.append(f"🔑 Palabras clave: {', '.join(category['palabras_clave'])}")
            report.append(f"⚡ Causas principales: {', '.join(category['causas_principales'])}")
            report.append("")
            report.append("📋 Ejemplos representativos:")
            
            for i, example in enumerate(category['ejemplos'][:3], 1):
                report.append(f"  {i}. ID: {example['id']}")
                report.append(f"     Resumen: {example['resumen']}")
                report.append(f"     Causa: {example['causa_raiz']}")
                report.append("")
            
            report.append("=" * 60)
            report.append("")
        
        return '\n'.join(report)
    
    def export_to_json(self, filename='incident_categories.json'):
        """Exporta las categorías a JSON"""
        if not self.categories:
            print("❌ No hay categorías para exportar")
            return
        
        # Preparar datos para JSON
        json_data = {
            'metadata': {
                'fecha_analisis': datetime.now().isoformat(),
                'total_categorias': len(self.categories),
                'max_categorias_permitidas': self.max_categories
            },
            'categorias': self.categories
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Categorías exportadas a {filename}")

def main():
    """Función principal para ejecutar el análisis"""
    print("🚀 Iniciando Analizador Semántico de Incidencias")
    
    # Cargar datos
    try:
        df = pd.read_csv('infomation.csv', encoding='ISO-8859-1', sep=';')
        print(f"✅ Datos cargados: {df.shape[0]} registros")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return
    
    # Filtrar registros válidos
    df_clean = df[['Ticket ID', 'Resumen', 'Notas', 'Causa Raiz']].copy()
    df_clean = df_clean.dropna(subset=['Causa Raiz'])
    print(f"✅ Registros válidos: {df_clean.shape[0]}")
    
    # Crear analizador
    analyzer = SemanticIncidentAnalyzer(max_categories=20)
    
    # Ejecutar análisis
    categories = analyzer.analyze_incidents(df_clean)
    
    if categories:
        # Generar y mostrar reporte
        report = analyzer.generate_report()
        print(report)
        
        # Exportar a JSON
        analyzer.export_to_json()
        
        # Guardar reporte en archivo
        with open('semantic_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print("✅ Reporte guardado en semantic_analysis_report.txt")
    
    else:
        print("❌ No se pudieron generar categorías")

if __name__ == "__main__":
    main()
