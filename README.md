# 🎬 Sistema de Análisis y Recomendación de Películas

## 📋 Descripción
Sistema completo de análisis y visualización de datos cinematográficos que proporciona insights detallados sobre películas, incluyendo análisis financiero, puntuaciones, tendencias temporales y un sistema de recomendación. Desarrollado con Python y Streamlit, ofrece una interfaz interactiva y amigable para explorar datos de películas.

### 🌟 Características Principales
- Dashboard interactivo con múltiples visualizaciones
- Análisis detallado por género y temporal
- Métricas financieras y de popularidad
- Sistema de recomendación personalizado
- Filtros dinámicos para análisis específicos
- Visualizaciones estadísticas avanzadas

## 🔧 Estructura del Proyecto
```
movie-analysis/
│
├── app_aux.py            # Punto de entrada principal
├── dashboard.py          # Implementación del dashboard
├── analysis.py           # Funciones de análisis de datos
├── visualization.py      # Funciones de visualización
├── requirements.txt      # Dependencias del proyecto
└── README.md            # Este archivo
```

## 📚 Requisitos del Sistema
- Python 3.8+
- Dependencias principales:
  ```
  streamlit>=1.20.0
  pandas>=1.5.0
  numpy>=1.23.0
  matplotlib>=3.6.0
  seaborn>=0.12.0
  plotly>=5.13.0
  scipy>=1.9.0
  scikit-learn>=1.2.0
  ```

## 🚀 Instalación y Uso

### Instalación
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/your-username/movie-analysis.git
   cd movie-analysis
   ```

2. Crear y activar un entorno virtual:
   ```bash
   python -m venv venv
   
   # En Windows:
   .\venv\Scripts\activate
   
   # En macOS/Linux:
   source venv/bin/activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Uso
1. Ejecutar la aplicación:
   ```bash
   streamlit run app_aux.py
   ```

2. Cargar datos:
   - Usar el botón "Cargar dataset de películas" en la barra lateral
   - Formatos soportados: CSV, XLSX
   - El dataset debe contener las siguientes columnas:
     - title: Título de la película
     - genre: Género
     - release_year: Año de estreno
     - critic_score: Puntuación de críticos (0-100)
     - audience_score: Puntuación de audiencia (0-100)
     - box_office_millions: Recaudación en millones
     - popularity: Índice de popularidad
     - duration_minutes: Duración en minutos
     - budget_millions: Presupuesto en millones

3. Explorar análisis:
   - Usar las pestañas para navegar entre diferentes análisis
   - Ajustar filtros en la barra lateral
   - Interactuar con las visualizaciones

## 📊 Funcionalidades Principales

### 1. Resumen General
- Métricas clave y KPIs
- Distribución de variables principales
- Análisis de outliers y calidad de datos

### 2. Análisis por Género
- Distribución de puntuaciones
- Tendencias de popularidad
- Análisis financiero por género

### 3. Análisis Financiero
- ROI y métricas de rentabilidad
- Análisis de presupuesto vs recaudación
- Tendencias financieras

### 4. Visualizaciones Interactivas
- Gráficos personalizables
- Análisis multivariable
- Comparativas detalladas

### 5. Evolución Temporal
- Tendencias históricas
- Análisis por década
- Patrones estacionales

### 6. Sistema de Recomendación
- Recomendaciones personalizadas
- Filtros por género y año
- Análisis de similitud

## 🤝 Contribución
Las contribuciones son bienvenidas. Para contribuir:

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request
