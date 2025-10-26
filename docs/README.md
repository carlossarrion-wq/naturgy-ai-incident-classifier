# Naturgy AI Incident Classifier

## Sistema Completo de Clasificación Automática de Incidencias

### 🎯 Descripción

Este proyecto implementa un sistema de Inteligencia Artificial para la **clasificación automática de incidencias del sistema Delta de Naturgy**. El sistema utiliza técnicas de Machine Learning y procesamiento de lenguaje natural para agrupar incidencias similares y entrenar modelos predictivos que permiten un triaje automático de nuevas incidencias.

### 🏗️ Arquitectura del Sistema

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Datos de       │    │  Preprocesamiento│    │  Extracción de  │
│  Incidencias    │───▶│  de Texto        │───▶│  Entidades      │
│  (Excel/CSV)    │    │  - Stop words    │    │  - CUPS         │
└─────────────────┘    │  - Normalización │    │  - Fechas       │
                       │  - Sinonimias    │    │  - Códigos      │
                       └──────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Clustering     │    │  Modelo          │    │  Clasificación  │
│  - K-Means      │◀───│  Predictivo      │───▶│  Automática     │
│  - Optimización │    │  - Random Forest │    │  - Triaje       │
│  - Validación   │    │  - Features      │    │  - Confianza    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### ✨ Características Principales

- **🧹 Preprocesamiento Inteligente**: Eliminación de stop words específicas de Naturgy y normalización de sinonimias
- **🔍 Extracción de Entidades**: Identificación automática de CUPS, códigos SR/REQ/OFL, fechas y otros elementos técnicos
- **🎯 Clustering Automático**: Agrupación de incidencias similares con optimización automática del número de clusters
- **🤖 Modelos Predictivos**: Random Forest y Gradient Boosting para clasificación de nuevas incidencias
- **📊 Evaluación Completa**: Métricas de validación y reportes detallados
- **🚀 Triaje en Producción**: API lista para integrar con sistemas ITSM como Remedy

### 🛠️ Instalación

#### Requisitos del Sistema
- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- openpyxl >= 3.0.0

#### Instalación de Dependencias

```bash
pip install pandas numpy scikit-learn openpyxl
```

#### Dependencias Opcionales

```bash
# Para funcionalidades avanzadas de NLP
pip install nltk

# Para integración con LLMs (requiere API keys)
pip install openai
```

### 🚀 Uso Rápido

#### 1. Entrenamiento del Modelo

```bash
python naturgy_ai_incident_classifier.py infomation.xlsx ./resultados
```

#### 2. Uso Programático

```python
from naturgy_ai_incident_classifier import NaturgyIncidentClassifier

# Entrenar modelo
classifier = NaturgyIncidentClassifier()
results = classifier.train_pipeline('datos_incidencias.xlsx')

# Clasificar nueva incidencia
prediction = classifier.classify_incident(
    "Error en CUPS ES0022000005514737AZ1P no permite calcular",
    additional_fields={'tipo_ticket': 'Incident'}
)

print(f"Tipo predicho: {prediction['predicted_type']}")
print(f"Confianza: {prediction['confidence']:.3f}")
```

#### 3. Ejecutar Pruebas

```bash
python test_naturgy_pipeline.py
```

### 📋 Estructura de Datos

El sistema espera un archivo Excel/CSV con las siguientes columnas:

| Columna | Descripción | Obligatorio |
|---------|-------------|-------------|
| `Ticket ID` | Identificador único del ticket | ✅ |
| `Resumen` | Resumen de la incidencia | ✅ |
| `Notas` | Descripción detallada | ✅ |
| `Tipo de Ticket` | Tipo (Incident, Service Request, etc.) | ✅ |

### 🎛️ Configuración

```python
config = {
    'max_clusters': 20,           # Máximo número de clusters
    'min_cluster_size': 5,        # Tamaño mínimo de cluster
    'tfidf_max_features': 5000,   # Features máximas TF-IDF
    'tfidf_min_df': 2,            # Frecuencia mínima de términos
    'tfidf_max_df': 0.8,          # Frecuencia máxima de términos
    'model_type': 'random_forest', # Tipo de modelo
    'cv_folds': 5                 # Pliegues para validación cruzada
}

classifier = NaturgyIncidentClassifier(config)
```

### 📊 Resultados del Análisis

El sistema genera varios archivos de salida:

1. **`naturgy_incident_model.pkl`**: Modelo entrenado serializado
2. **`naturgy_incident_analysis_complete.json`**: Análisis completo en formato JSON
3. **`naturgy_analysis_report.txt`**: Reporte legible para humanos
4. **`casos_prueba_naturgy.json`**: Casos de prueba generados automáticamente

### 🔧 Personalización

#### Stop Words Personalizadas

El sistema incluye stop words específicas de Naturgy:

```python
# Cortesía
'buenos días', 'cordial saludo', 'gracias'

# Sistema/metadatos
'delta', 'atlas', 'smile', 'sistema'

# Identificadores genéricos
'nisc', 'cliente', 'código', 'id'
```

#### Sinonimias y Normalización

```python
synonyms = {
    'fallo': 'error',
    'incidencia': 'error',
    'cancelación': 'baja',
    'activación': 'alta',
    # ... más sinonimias
}
```

#### Patrones de Entidades

```python
patterns = {
    'cups': r'ES\d{4}\d{4}\d{4}\d{4}[A-Z]{2}\d[A-Z]',
    'sr': r'R\d{2}-\d+',
    'req': r'REQ\d+',
    'ofl': r'OFL\d+',
    # ... más patrones
}
```

### 🔍 API de Clasificación

#### Clasificar Incidencia Individual

```python
resultado = classifier.classify_incident(
    incident_text="Error en sistema CUPS no permite modificar",
    additional_fields={
        'notas': 'Descripción adicional',
        'tipo_ticket': 'Incident'
    }
)

# Resultado
{
    'predicted_type': 'tipo_03',
    'confidence': 0.85,
    'type_info': {
        'nombre': 'Errores Técnicos CUPS',
        'descripcion': 'Agrupa incidencias relacionadas con...',
        'num_incidencias': 145
    },
    'extracted_entities': {
        'cups': ['ES0022000005514737AZ1P'],
        'fechas': [],
        'productos': []
    }
}
```

#### Cargar Modelo Pre-entrenado

```python
# Cargar modelo guardado
classifier = NaturgyIncidentClassifier.load_model('naturgy_incident_model.pkl')

# Usar directamente
prediction = classifier.classify_incident("Nueva incidencia...")
```

### 📈 Métricas y Evaluación

El sistema proporciona métricas completas:

- **Silhouette Score**: Calidad del clustering
- **Model Accuracy**: Precisión del modelo predictivo
- **Cross-Validation Score**: Validación cruzada
- **Distribución de Clusters**: Balanceado de tipos

### 🔄 Integración con Sistemas ITSM

#### Ejemplo de Integración con Remedy

```python
class RemedyIntegration:
    def __init__(self, classifier_model_path):
        self.classifier = NaturgyIncidentClassifier.load_model(classifier_model_path)
    
    def auto_triage_incident(self, incident_data):
        prediction = self.classifier.classify_incident(
            incident_data['summary'],
            {'tipo_ticket': incident_data['type']}
        )
        
        return {
            'assignment_group': prediction['type_info']['nombre'],
            'priority': self._calculate_priority(prediction),
            'confidence': prediction['confidence']
        }
```

### 🧪 Testing

#### Ejecutar Suite de Pruebas

```bash
# Pruebas básicas
python test_naturgy_pipeline.py

# Pruebas con datos reales (si disponibles)
python test_naturgy_pipeline.py --real-data

# Pruebas de rendimiento
python test_naturgy_pipeline.py --benchmark
```

#### Crear Casos de Prueba Personalizados

```python
def test_custom_incident():
    classifier = NaturgyIncidentClassifier()
    # ... entrenar modelo ...
    
    test_cases = [
        ("Error CUPS no permite facturar", "gestión_cups"),
        ("Solicito listado de clientes", "extracciones"),
        ("Fail batch job DECMD040P", "procesos_batch")
    ]
    
    for incident, expected_type in test_cases:
        prediction = classifier.classify_incident(incident)
        assert expected_type in prediction['predicted_type']
```

### 📚 Documentación Adicional

- [Manual de Usuario](docs/manual_usuario.md)
- [Guía de Desarrollo](docs/desarrollo.md)
- [API Reference](docs/api_reference.md)
- [Troubleshooting](docs/troubleshooting.md)

### 🤝 Contribución

1. Fork el proyecto
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

### 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

### 🆘 Soporte

Para soporte técnico:

1. Revisar [Troubleshooting](docs/troubleshooting.md)
2. Buscar en [Issues existentes](https://github.com/naturgy/incident-classifier/issues)
3. Crear nuevo issue con detalles completos
4. Contactar al equipo de desarrollo

---

**Desarrollado para Naturgy - Sistema Delta de Gestión de Incidencias**

*Versión 1.0.0 - Octubre 2025*
