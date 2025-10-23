#!/usr/bin/env python3
"""
Test script para verificar el funcionamiento del pipeline de clasificación de incidencias Naturgy
"""

import pandas as pd
import numpy as np
from naturgy_ai_incident_classifier import NaturgyIncidentClassifier, IncidentAnalysisRunner

def test_preprocessing():
    """Prueba las funciones de preprocesamiento"""
    print("🧪 Probando preprocesamiento...")
    
    classifier = NaturgyIncidentClassifier()
    
    # Datos de prueba
    test_data = pd.DataFrame({
        'Resumen': [
            'Buenos días, tengo un error en el sistema CUPS ES0022000005514737AZ1P',
            'Solicito extracción de listado de facturación',
            'Fail DECMD040P Job Name: DECMD040P Node: DELTAMAYPROD'
        ],
        'Notas': [
            'Por favor ayuda con este problema urgente',
            'Necesito el informe para el cliente',
            'Excessive batch process error'
        ],
        'Tipo de Ticket': [
            'Incident',
            'Service Request',
            'Incident'
        ]
    })
    
    # Procesar
    processed = classifier.preprocessor.process_dataframe(test_data)
    processed = classifier.entity_extractor.extract_entities(processed)
    
    print("✅ Texto original:", test_data['Resumen'].iloc[0])
    print("✅ Texto procesado:", processed['processed_text'].iloc[0])
    print("✅ Entidades extraídas:", processed['entities'].iloc[0])
    print("✅ Features de entidades:", [col for col in processed.columns if col.startswith('has_')])
    
    return processed

def test_clustering():
    """Prueba el clustering"""
    print("\n🧪 Probando clustering...")
    
    classifier = NaturgyIncidentClassifier()
    
    # Crear datos sintéticos más grandes para clustering
    test_texts = [
        'error sistema cups modificar baja',
        'error sistema cups modificar alta',
        'error sistema cups cambio fecha',
        'extracción informe listado facturación',
        'extracción informe listado consulta',
        'extracción informe datos cliente',
        'fail batch job proceso error',
        'fail batch job name error',
        'fail batch proceso sistema'
    ]
    
    test_data = pd.DataFrame({
        'Resumen': test_texts,
        'Notas': [''] * len(test_texts),
        'Tipo de Ticket': ['Incident'] * len(test_texts)
    })
    
    # Procesar y hacer clustering
    processed = classifier.preprocessor.process_dataframe(test_data)
    processed = classifier.entity_extractor.extract_entities(processed)
    
    cluster_results = classifier.clusterer.cluster_incidents(processed)
    
    print(f"✅ Clusters encontrados: {cluster_results['n_clusters']}")
    print(f"✅ Silhouette score: {cluster_results['silhouette_score']:.3f}")
    print(f"✅ Distribución de clusters: {cluster_results['cluster_sizes']}")
    
    return cluster_results

def test_classification():
    """Prueba la clasificación completa"""
    print("\n🧪 Probando clasificación completa...")
    
    # Usar datos sintéticos para prueba rápida
    test_data = pd.DataFrame({
        'Ticket ID': [f'TEST_{i:06d}' for i in range(20)],
        'Resumen': [
            'Error en CUPS ES0022000005514737AZ1P no permite modificar',
            'Solicito extracción de listado de facturación mensual',
            'Fail DECMD040P batch process error',
            'Problema con alta de contrato en sistema',
            'Consulta funcional sobre proceso de baja',
            'Error de conectividad en infraestructura',
            'Actualización masiva de datos pendiente',
            'Incidencia técnica en servidor principal',
            'Reporte de análisis de consumos requerido',
            'Modificación de fecha en línea de oferta',
            'Error en CUPS ES0234150261861527GB lectura',
            'Extracción de datos para auditoría necesaria',
            'Batch job DELED100P failed execution',
            'Cambio de titular sin subrogación requerido',
            'Consulta sobre operativa de facturación',
            'Error infraestructura comunicaciones red',
            'Actualización datos cliente origen usuario',
            'Problema técnico aplicación Delta',
            'Listado facturas conceptos importe cero',
            'Gestión CUPS direcciones incorrectas'
        ],
        'Notas': ['Descripción adicional del problema'] * 20,
        'Tipo de Ticket': ['Incident'] * 10 + ['Service Request'] * 10
    })
    
    classifier = NaturgyIncidentClassifier()
    
    try:
        # Guardar datos sintéticos y entrenar
        print("📊 Entrenando modelo con datos sintéticos...")
        test_data.to_excel('temp_test_data.xlsx', index=False)
        results = classifier.train_pipeline('temp_test_data.xlsx')
        
        # Limpiar archivo temporal
        import os
        if os.path.exists('temp_test_data.xlsx'):
            os.remove('temp_test_data.xlsx')
        
        # Probar clasificación de nuevas incidencias
        test_incidents = [
            "Error en sistema CUPS no permite calcular facturas",
            "Solicito informe de extracciones mensuales",
            "Fallo en proceso batch de datos masivos"
        ]
        
        print("\n🎯 Probando clasificación de nuevas incidencias:")
        for incident in test_incidents:
            prediction = classifier.classify_incident(incident)
            print(f"  📝 Incidencia: {incident}")
            print(f"  🏷️  Tipo predicho: {prediction['predicted_type']}")
            print(f"  🎯 Confianza: {prediction['confidence']:.3f}")
            print(f"  📋 Información: {prediction['type_info']['nombre'] if prediction['type_info'] else 'N/A'}")
            print()
        
        print("✅ Clasificación completada exitosamente")
        return results
        
    except Exception as e:
        print(f"❌ Error en clasificación: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_complete_pipeline():
    """Prueba el pipeline completo con archivo real si está disponible"""
    print("\n🧪 Probando pipeline completo...")
    
    try:
        # Intentar con archivo real
        import os
        if os.path.exists('infomation.xlsx'):
            print("📊 Usando datos reales de infomation.xlsx")
            IncidentAnalysisRunner.run_complete_analysis('infomation.xlsx', './test_results')
            print("✅ Pipeline completo ejecutado con datos reales")
        else:
            print("ℹ️  Archivo infomation.xlsx no encontrado, usando datos sintéticos")
            
            # Crear dataset sintético más grande
            synthetic_data = create_synthetic_dataset(100)
            synthetic_data.to_excel('test_synthetic_data.xlsx', index=False)
            
            IncidentAnalysisRunner.run_complete_analysis('test_synthetic_data.xlsx', './test_results')
            print("✅ Pipeline completo ejecutado con datos sintéticos")
            
    except Exception as e:
        print(f"❌ Error en pipeline completo: {e}")
        import traceback
        traceback.print_exc()

def create_synthetic_dataset(n_samples: int) -> pd.DataFrame:
    """Crea un dataset sintético para pruebas"""
    
    # Plantillas de incidencias por tipo
    templates = {
        'Error CUPS': [
            'Error en CUPS {cups} no permite {action}',
            'Problema con CUPS {cups} en proceso de {action}',
            'Incidencia técnica CUPS {cups} fallo {action}'
        ],
        'Extracción Datos': [
            'Solicito extracción de {report_type} para {period}',
            'Necesito listado de {report_type} del {period}',
            'Reporte de {report_type} requerido urgente'
        ],
        'Batch Process': [
            'Fail {job_name} batch process error',
            'Proceso {job_name} fallo en ejecución',
            'Job {job_name} terminated with errors'
        ],
        'Consulta Funcional': [
            'Consulta sobre proceso de {operation}',
            'Información necesaria sobre {operation}',
            'Duda operativa relacionada con {operation}'
        ]
    }
    
    # Generar datos
    data = []
    for i in range(n_samples):
        template_type = np.random.choice(list(templates.keys()))
        template = np.random.choice(templates[template_type])
        
        # Rellenar plantilla con datos sintéticos
        if 'cups' in template:
            cups = f'ES{np.random.randint(1000, 9999)}{np.random.randint(1000, 9999)}{np.random.randint(1000, 9999)}{np.random.randint(1000, 9999)}XX1P'
            action = np.random.choice(['calcular', 'modificar', 'activar', 'dar baja'])
            text = template.format(cups=cups, action=action)
        elif 'report_type' in template:
            report_type = np.random.choice(['facturas', 'consumos', 'clientes', 'contratos'])
            period = np.random.choice(['enero 2025', 'febrero 2025', 'último trimestre'])
            text = template.format(report_type=report_type, period=period)
        elif 'job_name' in template:
            job_name = np.random.choice(['DECMD040P', 'DELED100P', 'DETFD251P'])
            text = template.format(job_name=job_name)
        elif 'operation' in template:
            operation = np.random.choice(['facturación', 'cambio titular', 'activación contrato'])
            text = template.format(operation=operation)
        else:
            text = template
        
        data.append({
            'Ticket ID': f'SYNTH_{i:06d}',
            'Resumen': text,
            'Notas': f'Descripción adicional para incidencia {i}',
            'Tipo de Ticket': 'Incident' if np.random.random() > 0.3 else 'Service Request'
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("🚀 INICIANDO PRUEBAS DEL PIPELINE NATURGY")
    print("=" * 60)
    
    # Ejecutar todas las pruebas
    test_preprocessing()
    test_clustering()
    test_classification()
    test_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("✅ TODAS LAS PRUEBAS COMPLETADAS")
