#!/usr/bin/env python3
# Script para generar casos de prueba y clasificarlos
"""
Separa 100 registros como casos de prueba, entrena el modelo con el resto
y clasifica los casos de prueba para generar un reporte detallado
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
import os

# Importar las clases del clasificador refactorizado
sys.path.append('src')
from naturgy_classifier_refactored import NaturgyIncidentClassifier, OutputManager

class TestCaseGenerator:
    """Genera casos de prueba y los clasifica con el modelo entrenado"""
    
    def __init__(self, data_path: str, n_test_cases: int = 100, output_dir: str = "test_results"):
        self.data_path = data_path
        self.n_test_cases = n_test_cases
        self.output_manager = OutputManager(output_dir)
        self.test_cases = None
        self.train_data = None
        self.classifier = None
        self.results = []
    
    def load_and_split_data(self):
        """Carga los datos y separa casos de prueba del conjunto de entrenamiento"""
        print(f"📊 Cargando datos desde {self.data_path}...")
        
        try:
            df = pd.read_excel(self.data_path)
            print(f"✅ Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False
        
        # Filtrar registros con información mínima necesaria
        required_columns = ['Ticket ID', 'Resumen', 'Notas']
        available_columns = [col for col in required_columns if col in df.columns]
        
        if len(available_columns) < 2:
            print(f"❌ No se encontraron suficientes columnas requeridas: {available_columns}")
            return False
        
        # Filtrar registros válidos (con al menos Resumen)
        df_valid = df[df['Resumen'].notna() & (df['Resumen'] != '')].copy()
        print(f"📋 Registros válidos: {df_valid.shape[0]}")
        
        if len(df_valid) < self.n_test_cases + 500:  # Mínimo para entrenamiento
            print(f"❌ No hay suficientes registros para separar {self.n_test_cases} casos de prueba")
            return False
        
        # Separar casos de prueba aleatoriamente
        self.train_data, self.test_cases = train_test_split(
            df_valid, 
            test_size=self.n_test_cases, 
            random_state=42,
            shuffle=True
        )
        
        print(f"🎯 Casos de prueba separados: {len(self.test_cases)}")
        print(f"🏋️ Datos para entrenamiento: {len(self.train_data)}")
        
        # Guardar casos de prueba
        test_cases_file = self.output_manager.get_path('data', 'casos_prueba_original.xlsx')
        self.test_cases.to_excel(test_cases_file, index=False)
        print(f"💾 Casos de prueba guardados: {test_cases_file}")
        
        return True
    
    def train_classifier(self):
        """Entrena el clasificador con los datos de entrenamiento"""
        print("🚀 Iniciando entrenamiento del clasificador...")
        
        # Crear clasificador
        self.classifier = NaturgyIncidentClassifier(output_dir=str(self.output_manager.base_dir))
        
        # Guardar datos de entrenamiento temporalmente
        temp_train_file = 'temp_train_data.xlsx'
        self.train_data.to_excel(temp_train_file, index=False)
        
        try:
            # Entrenar con datos de entrenamiento
            training_results = self.classifier.train_pipeline(temp_train_file)
            print("✅ Entrenamiento completado exitosamente")
            
            # Limpiar archivo temporal
            os.remove(temp_train_file)
            
            return True
            
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {e}")
            # Limpiar archivo temporal en caso de error
            if os.path.exists(temp_train_file):
                os.remove(temp_train_file)
            return False
    
    def classify_test_cases(self):
        """Clasifica los casos de prueba con el modelo entrenado"""
        print(f"🔍 Clasificando {len(self.test_cases)} casos de prueba...")
        
        if not self.classifier or not self.classifier.is_trained:
            print("❌ El clasificador no está entrenado")
            return False
        
        # Cargar el mapeo de categorías desde el archivo JSON generado
        analysis_file = self.output_manager.get_path('data', 'analisis_completo_naturgy.json')
        category_mapping = {}
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
                tipos_incidencia = analysis_data.get('tipos_de_incidencia', {})
                # Crear mapeo de ID a nombre
                for idx, (key, info) in enumerate(tipos_incidencia.items()):
                    tipo_id = f"tipo_{idx:02d}"
                    category_mapping[tipo_id] = {
                        'nombre': info.get('nombre', key),
                        'descripcion': info.get('descripcion', 'Sin descripción'),
                        'criticidad': info.get('nivel_criticidad', 'No evaluada'),
                        'palabras_clave': info.get('palabras_clave', [])
                    }
                print(f"✅ Mapeo de categorías cargado: {len(category_mapping)} categorías")
        except Exception as e:
            print(f"⚠️ No se pudo cargar el mapeo de categorías: {e}")
            category_mapping = {}
        
        self.results = []
        
        for idx, row in self.test_cases.iterrows():
            try:
                # Preparar información adicional
                additional_fields = {
                    'notas': str(row.get('Notas', '')),
                    'tipo_ticket': str(row.get('Tipo de ticket', ''))
                }
                
                # Clasificar incidencia
                result = self.classifier.classify_incident(
                    str(row['Resumen']), 
                    additional_fields
                )
                
                # Usar el mapeo de categorías si está disponible
                predicted_type = result['predicted_type']
                
                # Verificar si es una categoría de reglas semánticas específicas
                semantic_categories = {
                    'error_aparato_montaje_desmontaje': 'Error Aparato Montaje/Desmontaje',
                    'gestion_bajas_altas_fechas': 'Gestión Bajas/Altas/Fechas',
                    'error_calculo_facturacion': 'Error Cálculo/Facturación',
                    'error_datos_cliente_contrato': 'Error Datos Cliente/Contrato',
                    'error_ofertas_contratacion': 'Error Ofertas/Contratación',
                    'solicitud_extraccion_informes': 'Solicitud Extracción/Informes',
                    'error_tecnico_batch_ftp': 'Error Técnico Batch/FTP',
                    'gestion_servicios_productos': 'Gestión Servicios/Productos',
                    'gestion_cobros_pagos': 'Gestión Cobros/Pagos'
                }
                
                if predicted_type in semantic_categories:
                    nombre_categoria = result['type_info'].get('nombre', semantic_categories[predicted_type])
                    descripcion_categoria = result['type_info'].get('descripcion', 'Clasificación automática por reglas semánticas')
                    criticidad = result['type_info'].get('nivel_criticidad', 'Media')
                    palabras_clave = result['type_info'].get('palabras_clave', ['clasificación automática'])[:5]
                elif predicted_type in category_mapping:
                    mapped_info = category_mapping[predicted_type]
                    nombre_categoria = mapped_info['nombre']
                    descripcion_categoria = mapped_info['descripcion']
                    criticidad = mapped_info['criticidad']
                    palabras_clave = mapped_info['palabras_clave'][:5]
                else:
                    # Fallback a la información del resultado
                    nombre_categoria = result['type_info'].get('nombre', predicted_type)
                    descripcion_categoria = result['type_info'].get('descripcion', 'Sin descripción disponible')
                    criticidad = result['type_info'].get('nivel_criticidad', 'No evaluada')
                    palabras_clave = result['type_info'].get('palabras_clave', [])[:5]
                
                # Preparar resultado
                test_result = {
                    'ticket_id': str(row.get('Ticket ID', f'idx_{idx}')),
                    'resumen_original': str(row['Resumen'])[:200] + '...' if len(str(row['Resumen'])) > 200 else str(row['Resumen']),
                    'tipo_ticket_original': str(row.get('Tipo de ticket', 'No especificado')),
                    'categoria_predicha': predicted_type,
                    'confianza': round(result['confidence'], 3),
                    'nombre_categoria': nombre_categoria,
                    'descripcion_categoria': descripcion_categoria,
                    'criticidad': criticidad,
                    'palabras_clave': palabras_clave
                }
                
                self.results.append(test_result)
                
            except Exception as e:
                print(f"⚠️ Error clasificando ticket {row.get('Ticket ID', idx)}: {e}")
                # Añadir resultado con error
                error_result = {
                    'ticket_id': str(row.get('Ticket ID', f'idx_{idx}')),
                    'resumen_original': str(row['Resumen'])[:200] + '...' if len(str(row['Resumen'])) > 200 else str(row['Resumen']),
                    'tipo_ticket_original': str(row.get('Tipo de ticket', 'No especificado')),
                    'categoria_predicha': 'ERROR',
                    'confianza': 0.0,
                    'nombre_categoria': 'Error en clasificación',
                    'descripcion_categoria': f'Error: {str(e)}',
                    'criticidad': 'No evaluada',
                    'palabras_clave': []
                }
                self.results.append(error_result)
        
        print(f"✅ Clasificación completada: {len(self.results)} resultados")
        return True
    
    def generate_detailed_report(self):
        """Genera reporte detallado de los casos de prueba"""
        if not self.results:
            print("❌ No hay resultados para generar el reporte")
            return
        
        # Generar reporte de texto
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("📊 REPORTE DE CASOS DE PRUEBA - CLASIFICADOR NATURGY")
        report_lines.append("=" * 100)
        report_lines.append(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total de casos de prueba: {len(self.results)}")
        report_lines.append(f"Casos clasificados exitosamente: {len([r for r in self.results if r['categoria_predicha'] != 'ERROR'])}")
        report_lines.append(f"Casos con error: {len([r for r in self.results if r['categoria_predicha'] == 'ERROR'])}")
        report_lines.append("")
        
        # Estadísticas de confianza
        confidences = [r['confianza'] for r in self.results if r['confianza'] > 0]
        if confidences:
            report_lines.append("📈 ESTADÍSTICAS DE CONFIANZA:")
            report_lines.append(f"• Confianza promedio: {np.mean(confidences):.3f}")
            report_lines.append(f"• Confianza mínima: {np.min(confidences):.3f}")
            report_lines.append(f"• Confianza máxima: {np.max(confidences):.3f}")
            report_lines.append(f"• Casos con alta confianza (>0.8): {len([c for c in confidences if c > 0.8])}")
            report_lines.append("")
        
        # Distribución por categorías predichas
        category_dist = {}
        for result in self.results:
            cat = result['nombre_categoria']
            category_dist[cat] = category_dist.get(cat, 0) + 1
        
        report_lines.append("📊 DISTRIBUCIÓN POR CATEGORÍAS PREDICHAS:")
        for cat, count in sorted(category_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.results)) * 100
            report_lines.append(f"• {cat}: {count} casos ({percentage:.1f}%)")
        report_lines.append("")
        
        # Casos individuales
        report_lines.append("🔍 DETALLE DE CASOS DE PRUEBA:")
        report_lines.append("=" * 100)
        
        for i, result in enumerate(self.results, 1):
            report_lines.append(f"\n📋 CASO #{i:03d}")
            report_lines.append("-" * 60)
            report_lines.append(f"ID Ticket: {result['ticket_id']}")
            report_lines.append(f"Resumen Original: {result['resumen_original']}")
            report_lines.append(f"Tipo Original: {result['tipo_ticket_original']}")
            report_lines.append("")
            report_lines.append(f"🎯 CLASIFICACIÓN PREDICHA:")
            report_lines.append(f"• Categoría: {result['nombre_categoria']}")
            report_lines.append(f"• ID Categoría: {result['categoria_predicha']}")
            report_lines.append(f"• Confianza: {result['confianza']:.3f}")
            report_lines.append(f"• Criticidad: {result['criticidad']}")
            report_lines.append(f"• Descripción: {result['descripcion_categoria']}")
            
            if result['palabras_clave']:
                report_lines.append(f"• Palabras clave: {', '.join(result['palabras_clave'])}")
            
            report_lines.append("-" * 60)
        
        # Guardar reporte de texto
        report_content = '\n'.join(report_lines)
        report_file = self.output_manager.save_report(report_content, 'casos_prueba_detallado.txt')
        
        # Guardar resultados en JSON
        json_data = {
            'metadata': {
                'fecha_analisis': datetime.now().isoformat(),
                'total_casos': len(self.results),
                'casos_exitosos': len([r for r in self.results if r['categoria_predicha'] != 'ERROR']),
                'casos_error': len([r for r in self.results if r['categoria_predicha'] == 'ERROR']),
                'confianza_promedio': np.mean(confidences) if confidences else 0.0
            },
            'distribuciones': {
                'por_categoria': category_dist,
                'por_confianza': {
                    'alta_confianza': len([c for c in confidences if c > 0.8]) if confidences else 0,
                    'media_confianza': len([c for c in confidences if 0.5 <= c <= 0.8]) if confidences else 0,
                    'baja_confianza': len([c for c in confidences if c < 0.5]) if confidences else 0
                }
            },
            'casos_prueba': self.results
        }
        
        json_file = self.output_manager.save_json_data(json_data, 'casos_prueba_resultados.json')
        
        print("✅ Reportes generados:")
        print(f"📄 Reporte detallado: {report_file}")
        print(f"📊 Datos JSON: {json_file}")
        
        return report_file, json_file
    
    def run_complete_test(self):
        """Ejecuta el proceso completo de generación y clasificación de casos de prueba"""
        print("🚀 INICIANDO GENERACIÓN DE CASOS DE PRUEBA")
        print("=" * 60)
        
        # 1. Cargar y separar datos
        if not self.load_and_split_data():
            return False
        
        # 2. Entrenar clasificador
        if not self.train_classifier():
            return False
        
        # 3. Clasificar casos de prueba
        if not self.classify_test_cases():
            return False
        
        # 4. Generar reportes
        report_files = self.generate_detailed_report()
        
        print("\n" + "=" * 60)
        print("✅ PROCESO COMPLETADO EXITOSAMENTE")
        print(f"📂 Resultados guardados en: {self.output_manager.base_dir}")
        print("📊 Archivos generados:")
        print(f"  - casos_prueba_original.xlsx (casos de prueba originales)")
        print(f"  - casos_prueba_detallado.txt (reporte detallado)")
        print(f"  - casos_prueba_resultados.json (resultados en JSON)")
        print(f"  - Modelo entrenado y análisis completo en subcarpetas")
        
        return True

def main():
    """Función principal"""
    if len(sys.argv) < 2:
        print("Uso: python test_classifier.py <archivo_datos.xlsx> [num_casos_prueba] [directorio_salida]")
        print("\nEjemplo:")
        print("  python test_classifier.py infomation.xlsx 100 test_results")
        sys.exit(1)
    
    data_path = sys.argv[1]
    n_test_cases = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "test_results"
    
    # Crear generador de casos de prueba
    generator = TestCaseGenerator(data_path, n_test_cases, output_dir)
    
    # Ejecutar proceso completo
    success = generator.run_complete_test()
    
    if not success:
        print("❌ El proceso falló. Revisa los errores anteriores.")
        sys.exit(1)

if __name__ == "__main__":
    main()
