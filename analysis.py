import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def cargar_y_limpiar_datos(df):
    """
    Función para cargar y limpiar el dataset de películas
    """
    # Convertir columnas a tipos apropiados
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    df['critic_score'] = pd.to_numeric(df['critic_score'], errors='coerce')
    df['audience_score'] = pd.to_numeric(df['audience_score'], errors='coerce')
    df['box_office_millions'] = pd.to_numeric(df['box_office_millions'], errors='coerce')
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
    df['duration_minutes'] = pd.to_numeric(df['duration_minutes'], errors='coerce')
    df['budget_millions'] = pd.to_numeric(df['budget_millions'], errors='coerce')
    
    # Crear variables derivadas
    df = crear_variables_derivadas(df)
    
    return df

def crear_variables_derivadas(df):
    """
    Crea nuevas variables derivadas para enriquecer el análisis
    """
    # Variables financieras
    df['roi'] = ((df['box_office_millions'] - df['budget_millions']) / 
                 df['budget_millions'] * 100).round(2)
    df['beneficio_neto'] = (df['box_office_millions'] - df['budget_millions']).round(2)
    df['ratio_presupuesto_boxoffice'] = (df['box_office_millions'] / df['budget_millions']).round(2)

    # Categorización financiera
    df['exito_financiero'] = pd.qcut(df['roi'], q=4, 
                                    labels=['Bajo', 'Medio', 'Alto', 'Muy Alto'])
    
    # Variables de duración
    df['categoria_duracion'] = pd.qcut(df['duration_minutes'], q=4,
                                     labels=['Corta', 'Media', 'Larga', 'Muy Larga'])
    
    # Variables de puntuación
    df['diferencia_scores'] = (df['critic_score'] - df['audience_score']).round(2)
    df['promedio_scores'] = ((df['critic_score'] + df['audience_score']) / 2).round(2)
    df['categoria_critica'] = pd.qcut(df['critic_score'], q=5,
                                    labels=['Muy Mala', 'Mala', 'Regular', 'Buena', 'Excelente'])
    
    # Variables temporales
    df['decada'] = (df['release_year'] // 10 * 10).astype(str) + 's'
    df['antiguedad'] = (2024 - df['release_year']).astype(int)
    
    # Variables de popularidad
    df['categoria_popularidad'] = pd.qcut(df['popularity'], q=5,
                                        labels=['Muy Baja', 'Baja', 'Media', 'Alta', 'Muy Alta'])
    
    # Variables de eficiencia
    df['ingreso_por_minuto'] = (df['box_office_millions'] / df['duration_minutes']).round(2)
    df['costo_por_minuto'] = (df['budget_millions'] / df['duration_minutes']).round(2)
    
    # Indicadores de éxito
    df['exito_critica'] = df['critic_score'] >= df['critic_score'].median()
    df['exito_audiencia'] = df['audience_score'] >= df['audience_score'].median()
    df['exito_comercial'] = df['box_office_millions'] >= df['box_office_millions'].median()
    
    # Score general
    df['score_general'] = (
        0.3 * (df['critic_score'] / df['critic_score'].max()) +
        0.3 * (df['audience_score'] / df['audience_score'].max()) +
        0.2 * (df['popularity'] / df['popularity'].max()) +
        0.2 * (df['roi'] / df['roi'].max())
    ).round(3)
    
    return df

def analizar_valores_faltantes(df):
    """
    Analiza y reporta valores faltantes en el dataset
    """
    valores_faltantes = df.isnull().sum()
    porcentaje_faltantes = (valores_faltantes / len(df)) * 100
    
    reporte_faltantes = pd.DataFrame({
        'Valores Faltantes': valores_faltantes,
        'Porcentaje Faltantes': porcentaje_faltantes
    })
    
    return reporte_faltantes

def detectar_outliers(df, columnas_numericas):
    """
    Detecta outliers usando el método IQR
    """
    outliers_info = {}
    
    for columna in columnas_numericas:
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)][columna]
        
        outliers_info[columna] = {
            'cantidad': len(outliers),
            'porcentaje': (len(outliers) / len(df)) * 100,
            'limite_inferior': limite_inferior,
            'limite_superior': limite_superior
        }
    
    return outliers_info

def generar_analisis_datos_problematicos(df):
    """
    Genera un análisis combinado de valores faltantes y outliers
    """
    # Análisis de valores faltantes
    valores_faltantes = df.isnull().sum()
    porcentaje_faltantes = (valores_faltantes / len(df)) * 100
    
    # Análisis de outliers
    outliers_info = {}
    columnas_numericas = [
        'critic_score', 'audience_score', 'box_office_millions',
        'popularity', 'duration_minutes', 'budget_millions', 'roi'
    ]
    
    for columna in columnas_numericas:
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)][columna]
        outliers_info[columna] = len(outliers)
    
    # Crear DataFrame combinado
    analisis_df = pd.DataFrame({
        'Valores Faltantes': valores_faltantes,
        'Porcentaje Faltantes (%)': porcentaje_faltantes.round(2),
        'Cantidad Outliers': pd.Series(outliers_info),
        'Porcentaje Outliers (%)': pd.Series(outliers_info) / len(df) * 100
    })
    
    # Limpiar el DataFrame
    analisis_df = analisis_df.round(2)
    # Solo mostrar filas que tienen valores faltantes o outliers
    analisis_df = analisis_df[
        (analisis_df['Valores Faltantes'] > 0) | 
        (analisis_df['Cantidad Outliers'] > 0)
    ].sort_values('Porcentaje Faltantes (%)', ascending=False)
    
    return analisis_df

def calcular_estadisticas_basicas(df):
    """
    Calcula estadísticas básicas para columnas numéricas originales y derivadas
    """
    columnas_numericas = [
        # Columnas originales
        'critic_score', 'audience_score', 'box_office_millions', 
        'popularity', 'duration_minutes', 'budget_millions',
        # Columnas derivadas
        'roi', 'beneficio_neto', 'ratio_presupuesto_boxoffice',
        'diferencia_scores', 'promedio_scores', 'ingreso_por_minuto',
        'costo_por_minuto', 'score_general'
    ]
    
    estadisticas = {}
    for columna in columnas_numericas:
        datos = df[columna].dropna().values
        estadisticas[columna] = {
            'media': np.mean(datos),
            'mediana': np.median(datos),
            'desv_std': np.std(datos),
            'min': np.min(datos),
            'max': np.max(datos),
            'curtosis': stats.kurtosis(datos),
            'asimetria': stats.skew(datos)
        }
    
    return estadisticas

def analizar_correlaciones(df):
    """
    Analiza correlaciones entre variables numéricas originales y derivadas
    """
    columnas_numericas = [
        # Columnas originales
        'critic_score', 'audience_score', 'box_office_millions', 
        'popularity', 'duration_minutes', 'budget_millions',
        # Columnas derivadas
        'roi', 'beneficio_neto', 'ratio_presupuesto_boxoffice',
        'diferencia_scores', 'promedio_scores', 'ingreso_por_minuto',
        'costo_por_minuto', 'score_general'
    ]
    
    return df[columnas_numericas].corr()

def analizar_variables_derivadas(df):
    """
    Genera un resumen estadístico de las variables derivadas
    """
    # Variables numéricas derivadas
    vars_numericas = [
        'roi', 'beneficio_neto', 'ratio_presupuesto_boxoffice',
        'diferencia_scores', 'promedio_scores', 'ingreso_por_minuto',
        'costo_por_minuto', 'score_general'
    ]
    
    resumen_numerico = df[vars_numericas].describe().round(2)
    
    # Variables categóricas
    vars_categoricas = [
        'exito_financiero', 'categoria_duracion', 'categoria_critica',
        'decada', 'categoria_popularidad'
    ]
    
    resumen_categorico = {var: df[var].value_counts() for var in vars_categoricas}
    
    # Variables booleanas
    vars_booleanas = ['exito_critica', 'exito_audiencia', 'exito_comercial']
    resumen_booleano = {var: df[var].value_counts(normalize=True) * 100 
                        for var in vars_booleanas}
    
    return {
        'resumen_numerico': resumen_numerico,
        'resumen_categorico': resumen_categorico,
        'resumen_booleano': resumen_booleano
    }

def realizar_agregaciones(df):
    """
    Realiza múltiples agregaciones sobre los datos de películas
    """
    # 1. Agregaciones por Año
    agg_por_año = df.groupby('release_year').agg({
        'movie_id': 'count',
        'box_office_millions': ['sum', 'mean'],
        'budget_millions': ['sum', 'mean'],
        'critic_score': 'mean',
        'audience_score': 'mean',
        'popularity': 'mean',
        'roi': 'mean',
        'beneficio_neto': 'sum',
        'score_general': 'mean'
    }).round(2)
    agg_por_año.columns = ['num_peliculas', 'box_office_total', 'box_office_promedio', 
                          'budget_total', 'budget_promedio', 'critic_score_promedio',
                          'audience_score_promedio', 'popularidad_promedio', 'roi_promedio',
                          'beneficio_neto_total', 'score_general_promedio']

    # 2. Agregaciones por Década
    agg_por_decada = df.groupby('decada').agg({
        'movie_id': 'count',
        'box_office_millions': ['sum', 'mean'],
        'budget_millions': ['sum', 'mean'],
        'roi': ['mean', 'median'],
        'beneficio_neto': ['sum', 'mean'],
        'score_general': ['mean', 'median']
    }).round(2)

    # 3. Agregaciones por Género
    agg_por_genero = df.groupby('genre').agg({
        'movie_id': 'count',
        'box_office_millions': ['sum', 'mean'],
        'budget_millions': ['sum', 'mean'],
        'critic_score': 'mean',
        'audience_score': 'mean',
        'popularity': 'mean',
        'roi': 'mean',
        'duration_minutes': 'mean'
    }).round(2)

    # 4. Agregaciones por Categoría de Duración
    agg_por_duracion = df.groupby('categoria_duracion').agg({
        'movie_id': 'count',
        'box_office_millions': ['mean', 'sum'],
        'budget_millions': ['mean', 'sum'],
        'roi': 'mean',
        'critic_score': 'mean',
        'audience_score': 'mean',
        'popularity': 'mean'
    }).round(2)

    # 5. Agregaciones por Categoría de Éxito Financiero
    agg_por_exito = df.groupby('exito_financiero').agg({
        'movie_id': 'count',
        'box_office_millions': ['mean', 'sum'],
        'budget_millions': ['mean', 'sum'],
        'critic_score': 'mean',
        'audience_score': 'mean',
        'popularity': 'mean',
        'duration_minutes': 'mean'
    }).round(2)

    # 6. Agregaciones Cruzadas (Género x Década)
    agg_genero_decada = df.pivot_table(
        values=['box_office_millions', 'roi', 'score_general'],
        index='genre',
        columns='decada',
        aggfunc='mean'
    ).round(2)

    # 7. Top Películas por Diferentes Métricas
    top_movies = {
        'top_box_office': df.nlargest(10, 'box_office_millions')[
            ['title', 'release_year', 'box_office_millions', 'genre']],
        'top_roi': df.nlargest(10, 'roi')[
            ['title', 'release_year', 'roi', 'genre']],
        'top_critic_score': df.nlargest(10, 'critic_score')[
            ['title', 'release_year', 'critic_score', 'genre']],
        'top_audience_score': df.nlargest(10, 'audience_score')[
            ['title', 'release_year', 'audience_score', 'genre']],
        'top_score_general': df.nlargest(10, 'score_general')[
            ['title', 'release_year', 'score_general', 'genre']]
    }

    # 8. Estadísticas de Rentabilidad por Género y Década
    rentabilidad = df.pivot_table(
        values=['roi', 'beneficio_neto'],
        index='genre',
        columns='decada',
        aggfunc=['mean', 'sum']
    ).round(2)

    # 9. Análisis de Tendencias Temporales
    tendencias = df.groupby('release_year').agg({
        'critic_score': ['mean', 'std'],
        'audience_score': ['mean', 'std'],
        'roi': ['mean', 'std'],
        'score_general': ['mean', 'std']
    }).round(2)

    # 10. Métricas de Eficiencia
    eficiencia = df.groupby('genre').agg({
        'ingreso_por_minuto': ['mean', 'max'],
        'costo_por_minuto': ['mean', 'max'],
        'ratio_presupuesto_boxoffice': ['mean', 'max']
    }).round(2)

    return {
        'agregaciones_año': agg_por_año,
        'agregaciones_decada': agg_por_decada,
        'agregaciones_genero': agg_por_genero,
        'agregaciones_duracion': agg_por_duracion,
        'agregaciones_exito': agg_por_exito,
        'agregaciones_genero_decada': agg_genero_decada,
        'top_peliculas': top_movies,
        'rentabilidad_genero_decada': rentabilidad,
        'tendencias_temporales': tendencias,
        'metricas_eficiencia': eficiencia
    }

def analizar_distribuciones_por_grupo(df):
    """
    Analiza las distribuciones de métricas clave por diferentes grupos
    """
    metricas = ['box_office_millions', 'roi', 'critic_score', 'audience_score', 
                'popularity', 'score_general']
    grupos = ['genre', 'decada', 'categoria_duracion', 'exito_financiero']
    
    distribuciones = {}
    for grupo in grupos:
        distribuciones[grupo] = {}
        for metrica in metricas:
            distribuciones[grupo][metrica] = df.groupby(grupo)[metrica].describe().round(2)
    
    return distribuciones

def calcular_metricas_comparativas(df):
    """
    Calcula métricas comparativas entre diferentes grupos
    """
    # Comparación con promedios generales
    promedios_generales = df[['box_office_millions', 'roi', 'critic_score', 
                             'audience_score', 'popularity']].mean()
    
    comparativas = {}
    
    # Por género
    comparativas['genero'] = df.groupby('genre').agg({
        'box_office_millions': lambda x: (x.mean() / promedios_generales['box_office_millions'] - 1) * 100,
        'roi': lambda x: (x.mean() / promedios_generales['roi'] - 1) * 100,
        'critic_score': lambda x: (x.mean() / promedios_generales['critic_score'] - 1) * 100,
        'audience_score': lambda x: (x.mean() / promedios_generales['audience_score'] - 1) * 100,
        'popularity': lambda x: (x.mean() / promedios_generales['popularity'] - 1) * 100
    }).round(2)
    
    # Por década
    comparativas['decada'] = df.groupby('decada').agg({
        'box_office_millions': lambda x: (x.mean() / promedios_generales['box_office_millions'] - 1) * 100,
        'roi': lambda x: (x.mean() / promedios_generales['roi'] - 1) * 100,
        'critic_score': lambda x: (x.mean() / promedios_generales['critic_score'] - 1) * 100,
        'audience_score': lambda x: (x.mean() / promedios_generales['audience_score'] - 1) * 100,
        'popularity': lambda x: (x.mean() / promedios_generales['popularity'] - 1) * 100
    }).round(2)
    
    return comparativas

def generar_reporte_agregaciones(df):
    """
    Genera un reporte completo de todas las agregaciones
    """
    # Realizar todas las agregaciones
    agregaciones = realizar_agregaciones(df)
    distribuciones = analizar_distribuciones_por_grupo(df)
    comparativas = calcular_metricas_comparativas(df)
    
    # Imprimir resumen
    print("1. Resumen por Año:")
    print(agregaciones['agregaciones_año'].head())
    
    print("\n2. Resumen por Década:")
    print(agregaciones['agregaciones_decada'])
    
    print("\n3. Resumen por Género:")
    print(agregaciones['agregaciones_genero'])
    
    print("\n4. Top 5 Películas por Box Office:")
    print(agregaciones['top_peliculas']['top_box_office'].head())
    
    print("\n5. Comparativas por Género (% diferencia con promedio general):")
    print(comparativas['genero'])
    
    return {
        'agregaciones': agregaciones,
        'distribuciones': distribuciones,
        'comparativas': comparativas
    }

def generar_resumen_estadistico(df):
    """
    Genera un resumen estadístico detallado del dataset de películas
    """
    # Seleccionar columnas numéricas relevantes
    columnas_numericas = {
        'Puntuación Críticos': 'critic_score',
        'Puntuación Audiencia': 'audience_score',
        'Recaudación (M$)': 'box_office_millions',
        'Presupuesto (M$)': 'budget_millions',
        'Popularidad': 'popularity',
        'Duración (min)': 'duration_minutes',
        'ROI (%)': 'roi'
    }
    
    # Crear DataFrame con estadísticas
    stats_dict = {}
    for nombre, columna in columnas_numericas.items():
        stats = df[columna].describe()
        stats_dict[nombre] = {
            'Media': round(stats['mean'], 2),
            'Mediana': round(stats['50%'], 2),
            'Desv. Est.': round(stats['std'], 2),
            'Mínimo': round(stats['min'], 2),
            'Máximo': round(stats['max'], 2),
            'Q1': round(stats['25%'], 2),
            'Q3': round(stats['75%'], 2)
        }
    
    return pd.DataFrame(stats_dict).T

def calcular_metricas_adicionales(df):
    """
    Calcula métricas adicionales incluyendo las mejores y peores películas en cada categoría
    """
    metricas = {
        # Rentabilidad
        'Películas más rentables': df.nlargest(5, 'roi')[
            ['title', 'release_year', 'roi', 'box_office_millions']
        ].assign(roi=lambda x: x['roi'].round(2)),
        
        'Películas menos rentables': df.nsmallest(5, 'roi')[
            ['title', 'release_year', 'roi', 'box_office_millions']
        ].assign(roi=lambda x: x['roi'].round(2)),
        
        # Presupuesto
        'Mayor presupuesto': df.nlargest(5, 'budget_millions')[
            ['title', 'release_year', 'budget_millions', 'box_office_millions']
        ],
        
        'Menor presupuesto': df.nsmallest(5, 'budget_millions')[
            ['title', 'release_year', 'budget_millions', 'box_office_millions']
        ],
        
        # Puntuaciones críticos
        'Mejor puntuadas por críticos': df.nlargest(5, 'critic_score')[
            ['title', 'release_year', 'critic_score', 'audience_score']
        ].assign(
            critic_score=lambda x: x['critic_score'].round(1),
            audience_score=lambda x: x['audience_score'].round(1)
        ),
        
        'Peor puntuadas por críticos': df.nsmallest(5, 'critic_score')[
            ['title', 'release_year', 'critic_score', 'audience_score']
        ].assign(
            critic_score=lambda x: x['critic_score'].round(1),
            audience_score=lambda x: x['audience_score'].round(1)
        ),
        
        # Puntuaciones audiencia
        'Mejor puntuadas por audiencia': df.nlargest(5, 'audience_score')[
            ['title', 'release_year', 'audience_score', 'critic_score']
        ].assign(
            critic_score=lambda x: x['critic_score'].round(1),
            audience_score=lambda x: x['audience_score'].round(1)
        ),
        
        'Peor puntuadas por audiencia': df.nsmallest(5, 'audience_score')[
            ['title', 'release_year', 'audience_score', 'critic_score']
        ].assign(
            critic_score=lambda x: x['critic_score'].round(1),
            audience_score=lambda x: x['audience_score'].round(1)
        ),
        
        # Popularidad
        'Películas más populares': df.nlargest(5, 'popularity')[
            ['title', 'release_year', 'popularity', 'box_office_millions']
        ].assign(popularity=lambda x: x['popularity'].round(1)),
        
        'Películas menos populares': df.nsmallest(5, 'popularity')[
            ['title', 'release_year', 'popularity', 'box_office_millions']
        ].assign(popularity=lambda x: x['popularity'].round(1)),
        
        # Estadísticas por género
        'Géneros más comunes': df['genre'].value_counts().head(),
        'Duración promedio por género': df.groupby('genre')['duration_minutes'].mean().round(1)
    }
    
    return metricas

def analisis_completo(df):
    """
    Realiza un análisis completo del dataset de películas incluyendo variables derivadas
    """
    print("1. Limpieza inicial de datos y creación de variables derivadas")
    df_procesado = cargar_y_limpiar_datos(df)
    
    print("\n2. Análisis de valores faltantes:")
    reporte_faltantes = analizar_valores_faltantes(df_procesado)
    print(reporte_faltantes)
    
    print("\n3. Detección de outliers:")
    columnas_numericas = [
        'critic_score', 'audience_score', 'box_office_millions', 
        'popularity', 'duration_minutes', 'budget_millions',
        'roi', 'beneficio_neto', 'ratio_presupuesto_boxoffice'
    ]
    outliers = detectar_outliers(df_procesado, columnas_numericas)
    for columna, info in outliers.items():
        print(f"\n{columna}:")
        print(f"Cantidad de outliers: {info['cantidad']}")
        print(f"Porcentaje de outliers: {info['porcentaje']:.2f}%")
    
    print("\n4. Estadísticas básicas:")
    estadisticas = calcular_estadisticas_basicas(df_procesado)
    for columna, stats in estadisticas.items():
        print(f"\n{columna}:")
        for metric, valor in stats.items():
            print(f"{metric}: {valor:.2f}")
    
    print("\n5. Matriz de correlaciones:")
    correlaciones = analizar_correlaciones(df_procesado)
    print(correlaciones)
    
    print("\n6. Análisis de variables derivadas:")
    analisis_derivadas = analizar_variables_derivadas(df_procesado)
    print("\nResumen de variables numéricas derivadas:")
    print(analisis_derivadas['resumen_numerico'])
    print("\nResumen de variables categóricas:")
    for var, counts in analisis_derivadas['resumen_categorico'].items():
        print(f"\n{var}:")
        print(counts)
    print("\nResumen de indicadores de éxito (%):")
    for var, porcentajes in analisis_derivadas['resumen_booleano'].items():
        print(f"\n{var}:")
        print(porcentajes)

    # Nuevas secciones de agregaciones
    print("\n7. Agregaciones por Año:")
    agregaciones = realizar_agregaciones(df_procesado)
    print(agregaciones['agregaciones_año'].head())
    
    print("\n8. Agregaciones por Género:")
    print(agregaciones['agregaciones_genero'])
    
    print("\n9. Top 10 Películas:")
    print("\nPor Box Office:")
    print(agregaciones['top_peliculas']['top_box_office'])
    print("\nPor ROI:")
    print(agregaciones['top_peliculas']['top_roi'])
    
    print("\n10. Análisis de Distribuciones:")
    distribuciones = analizar_distribuciones_por_grupo(df_procesado)
    for grupo, metricas in distribuciones.items():
        print(f"\nDistribuciones para {grupo}:")
        for metrica, stats in metricas.items():
            print(f"\n{metrica}:")
            print(stats)
    
    print("\n11. Métricas Comparativas:")
    comparativas = calcular_metricas_comparativas(df_procesado)
    print("\nComparativas por Género:")
    print(comparativas['genero'])
    print("\nComparativas por Década:")
    print(comparativas['decada'])

    return {
        'df_procesado': df_procesado,
        'valores_faltantes': reporte_faltantes,
        'outliers': outliers,
        'estadisticas': estadisticas,
        'correlaciones': correlaciones,
        'analisis_derivadas': analisis_derivadas,
        'agregaciones': {
            'por_año': agregaciones['agregaciones_año'],
            'por_genero': agregaciones['agregaciones_genero'],
            'por_decada': agregaciones['agregaciones_decada'],
            'por_duracion': agregaciones['agregaciones_duracion'],
            'por_exito': agregaciones['agregaciones_exito'],
            'genero_decada': agregaciones['agregaciones_genero_decada'],
            'top_peliculas': agregaciones['top_peliculas'],
            'rentabilidad': agregaciones['rentabilidad_genero_decada'],
            'tendencias': agregaciones['tendencias_temporales'],
            'eficiencia': agregaciones['metricas_eficiencia']
        },
        'distribuciones': distribuciones,
        'comparativas': comparativas
    }

# Añadir estas funciones junto con las demás funciones de utilidad:
def crear_sistema_recomendacion(df):
    """
    Crea un sistema de recomendación usando K-Means
    """
    features = ['release_year', 'popularity', 'critic_score', 'audience_score', 
                'box_office_millions', 'budget_millions', 'roi']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    
    n_clusters = min(8, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    
    return scaler, kmeans, features

def recomendar_peliculas(df, genero, min_year, popularidad_min, n_recomendaciones=5):
    """
    Sistema de recomendación simplificado y robusto
    """
    # Primer intento con filtros originales
    df_filtered = df[
        (df['genre'] == genero) &
        (df['release_year'] >= min_year) &
        (df['popularity'] >= popularidad_min)
    ].copy()
    
    filtros_relajados = []
    
    # Si no hay suficientes resultados, relajar filtros gradualmente
    if len(df_filtered) < n_recomendaciones:
        # Primer intento: relajar popularidad
        df_filtered = df[
            (df['genre'] == genero) &
            (df['release_year'] >= min_year) &
            (df['popularity'] >= popularidad_min * 0.5)
        ].copy()
        if len(df_filtered) >= n_recomendaciones:
            filtros_relajados.append("popularidad")
    
    # Si aún no hay suficientes, relajar año
    if len(df_filtered) < n_recomendaciones:
        df_filtered = df[
            (df['genre'] == genero) &
            (df['release_year'] >= min_year - 10) &
            (df['popularity'] >= popularidad_min * 0.5)
        ].copy()
        if len(df_filtered) >= n_recomendaciones:
            filtros_relajados.append("año")
    
    # Si aún no hay suficientes, considerar géneros similares
    if len(df_filtered) < n_recomendaciones:
        generos_similares = {
            'Action': ['Thriller', 'Sci-Fi'],
            'Comedy': ['Romance', 'Animation'],
            'Drama': ['Thriller', 'Romance'],
            'Thriller': ['Action', 'Drama'],
            'Sci-Fi': ['Action', 'Thriller'],
            'Romance': ['Comedy', 'Drama'],
            'Animation': ['Comedy', 'Fantasy'],
            'Fantasy': ['Animation', 'Sci-Fi']
        }
        
        generos_a_buscar = generos_similares.get(genero, [])
        if generos_a_buscar:
            df_filtered = df[
                (df['genre'].isin([genero] + generos_a_buscar)) &
                (df['release_year'] >= min_year - 10) &
                (df['popularity'] >= popularidad_min * 0.5)
            ].copy()
            if len(df_filtered) >= n_recomendaciones:
                filtros_relajados.append("género")
    
    # Si aún no hay resultados, usar solo el género como filtro
    if len(df_filtered) < n_recomendaciones:
        df_filtered = df[df['genre'] == genero].copy()
        filtros_relajados = ["todos los criterios"]
    
    if len(df_filtered) == 0:
        return None, None
    
    # Calcular score de recomendación
    df_filtered['score'] = (
        df_filtered['popularity'] * 0.4 +
        df_filtered['audience_score'] * 0.3 +
        df_filtered['critic_score'] * 0.3
    )
    
    return df_filtered.nlargest(n_recomendaciones, 'score'), filtros_relajados