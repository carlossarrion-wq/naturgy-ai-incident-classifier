# 🚀 Naturgy AI Incident Classifier

Sistema de Inteligencia Artificial para clasificación automática de incidencias del sistema Delta de Naturgy.

## 📋 Descripción del Proyecto

Este proyecto desarrolla un sistema completo de IA que automatiza la clasificación de incidencias mediante:

- **Análisis automático de texto** con preprocesamiento específico para Naturgy
- **Clustering inteligente** que identifica 25 tipos específicos de incidencias
- **Modelo predictivo** con 90.7% de precisión para clasificación automática
- **API lista para producción** e integración con ITSM Remedy

## 🎯 Resultados Principales

- ✅ **25 tipos específicos** de incidencias identificados automáticamente
- ✅ **90.7% de precisión** en clasificación automática
- ✅ **3,943 incidencias** procesadas y analizadas
- ✅ **Sistema listo** para producción

## 📁 Estructura del Proyecto

```
Naturgy/
├── src/                          # Código fuente principal
│   ├── naturgy_classifier.py     # Sistema principal de clasificación
│   └── semantic_analyzer.py      # Analizador semántico adicional
├── docs/                         # Documentación
│   ├── README.md                 # Documentación técnica detallada
│   ├── resumen_ejecutivo_proyecto_naturgy.txt    # Resumen para management
│   └── explicacion_centroides_naturgy.txt        # Explicación técnica de centroides
├── models/                       # Modelos entrenados
│   └── naturgy_incident_model.pkl               # Modelo principal entrenado
├── outputs/                      # Resultados y reportes
│   ├── naturgy_incident_analysis_complete.json  # Análisis completo
│   ├── naturgy_analysis_report.txt              # Reporte legible
│   ├── casos_prueba_naturgy.json                # Casos de prueba
│   └── centroids_analysis_naturgy.json          # Análisis de centroides
├── tests/                        # Scripts de prueba y análisis
│   ├── test_naturgy_pipeline.py  # Tests del pipeline completo
│   └── extract_centroids_demo.py # Demo de extracción de centroides
└── infomation.xlsx              # Dataset original de incidencias
```

## 🚀 Uso Rápido

### Entrenamiento del Modelo

```bash
python3 src/naturgy_classifier.py infomation.xlsx
```

### Análisis de Centroides

```bash
python3 tests/extract_centroids_demo.py
```

### Clasificación de Nueva Incidencia

```python
from src.naturgy_classifier import NaturgyIncidentClassifier

# Cargar modelo entrenado
classifier = NaturgyIncidentClassifier.load_model('models/naturgy_incident_model.pkl')

# Clasificar nueva incidencia
result = classifier.classify_incident("Error al calcular factura en sistema Delta")

print(f"Tipo predicho: {result['predicted_type']}")
print(f"Confianza: {result['confidence']:.2%}")
```

## 📊 Tipos de Incidencias Identificados

Los 25 tipos principales incluyen:

1. **ERROR DELTA FACTURAS** (849 casos - 21.5%)
2. **ATLAS ERROR OFERTA** (636 casos - 16.1%)
3. **ERROR APARATO ALTA** (346 casos - 8.8%)
4. **BAJA ATLAS FECHA** (265 casos - 6.7%)
5. **LISTADO INFORME CLIENTES** (187 casos - 4.7%)
6. **ERROR CALCULAR CUPS** (183 casos - 4.6%)

*[Ver lista completa en `outputs/naturgy_analysis_report.txt`]*

## ⚙️ Requisitos Técnicos

### Dependencias Python

```bash
pip install pandas numpy scikit-learn openpyxl
```

### Dependencias Opcionales

```bash
pip install nltk  # Para procesamiento avanzado de texto
```

## 🔧 Arquitectura Técnica

### Componentes Principales

1. **TextPreprocessor**: Limpieza y normalización específica para Naturgy
2. **EntityExtractor**: Extracción de CUPS, códigos, fechas y entidades técnicas
3. **IncidentClusterer**: Motor de clustering K-Means optimizado
4. **PredictiveClassifier**: Modelo Random Forest para clasificación

### Flujo de Procesamiento

```
Datos Brutos → Preprocesamiento → Extracción Entidades → 
Clustering → Entrenamiento Modelo → Clasificación Automática
```

## 📈 Métricas de Performance

- **Accuracy del Modelo**: 90.7%
- **Método de Validación**: Train/Test Split (80/20) + Cross-Validation 5-fold
- **Silhouette Score**: 0.085
- **Número de Clusters**: 25
- **Vocabulario TF-IDF**: 8,000 términos

## 🎯 Integración en Producción

### API de Clasificación

El sistema está diseñado para integrarse directamente con ITSM Remedy:

```python
# Ejemplo de integración
def classify_remedy_ticket(ticket_data):
    classifier = NaturgyIncidentClassifier.load_model()
    
    result = classifier.classify_incident(
        incident_text=ticket_data['summary'],
        additional_fields={
            'notas': ticket_data.get('notes', ''),
            'tipo_ticket': ticket_data.get('type', '')
        }
    )
    
    return {
        'incident_type': result['predicted_type'],
        'confidence': result['confidence'],
        'suggested_group': get_support_group(result['predicted_type'])
    }
```

### Monitoreo en Producción

- **Re-entrenamiento**: Mensual con nuevos datos
- **Métricas de seguimiento**: Precisión semanal, distribución de confianzas
- **Umbrales de alerta**: Precisión < 85% para revisión
 

## 📄 Documentación Adicional

- **Resumen Ejecutivo**: `docs/resumen_ejecutivo_proyecto_naturgy.txt`
- **Análisis Técnico Centroides**: `docs/explicacion_centroides_naturgy.txt`
- **Reporte Detallado**: `outputs/naturgy_analysis_report.txt`
- **Documentación Técnica**: `docs/README.md`

---

**🏆 Logros del Proyecto**:  
Sistema IA que transforma completamente el triaje de 4,000 incidencias/mes con 90.7% de precisión, habilitando ahorro de 2-3 FTE y mejora significativa en SLAs de resolución.
