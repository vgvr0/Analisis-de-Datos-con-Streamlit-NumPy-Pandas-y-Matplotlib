import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd 

def configurar_estilo_plots():
    """
    Configura el estilo general de las visualizaciones usando estilos válidos de matplotlib
    """
    plt.style.use('fivethirtyeight')  # Cambiado de 'seaborn' a 'fivethirtyeight'
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

def plot_distribucion_puntuaciones(df):
    """
    Crea visualizaciones de la distribución de puntuaciones (críticos y audiencia)
    """
    configurar_estilo_plots()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Histograma de puntuaciones de críticos
    sns.histplot(data=df, x='critic_score', bins=30, ax=ax1, color='skyblue')
    ax1.set_title('Distribución de Puntuaciones de Críticos')
    ax1.set_xlabel('Puntuación')
    ax1.set_ylabel('Frecuencia')
    
    # Añadir línea de densidad y estadísticas
    critic_mean = df['critic_score'].mean()
    critic_median = df['critic_score'].median()
    ax1.axvline(critic_mean, color='red', linestyle='--', alpha=0.5, 
                label=f'Media: {critic_mean:.1f}')
    ax1.axvline(critic_median, color='green', linestyle='--', alpha=0.5, 
                label=f'Mediana: {critic_median:.1f}')
    ax1.legend()
    
    # 2. Histograma de puntuaciones de audiencia
    sns.histplot(data=df, x='audience_score', bins=30, ax=ax2, color='lightgreen')
    ax2.set_title('Distribución de Puntuaciones de Audiencia')
    ax2.set_xlabel('Puntuación')
    ax2.set_ylabel('Frecuencia')
    
    # Añadir línea de densidad y estadísticas
    audience_mean = df['audience_score'].mean()
    audience_median = df['audience_score'].median()
    ax2.axvline(audience_mean, color='red', linestyle='--', alpha=0.5, 
                label=f'Media: {audience_mean:.1f}')
    ax2.axvline(audience_median, color='green', linestyle='--', alpha=0.5, 
                label=f'Mediana: {audience_median:.1f}')
    ax2.legend()
    
    # 3. Box plot comparativo
    df_melted = df.melt(value_vars=['critic_score', 'audience_score'], 
                        var_name='Tipo', value_name='Puntuación')
    sns.boxplot(data=df_melted, x='Tipo', y='Puntuación', ax=ax3)
    ax3.set_title('Comparación de Distribuciones (Box Plot)')
    
    # 4. Violin plot comparativo
    sns.violinplot(data=df_melted, x='Tipo', y='Puntuación', ax=ax4)
    ax4.set_title('Comparación de Distribuciones (Violin Plot)')
    
    plt.tight_layout(pad=3.0)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    return fig

def plot_distribucion_por_genero(df):
    """
    Visualiza la distribución de puntuaciones de críticos y audiencia por género en un solo gráfico
    """
    configurar_estilo_plots()
    
    # Preparar los datos para el gráfico
    # Convertir los datos a formato largo para facilitar el plotting
    df_melted = pd.melt(
        df,
        id_vars=['genre'],
        value_vars=['critic_score', 'audience_score'],
        var_name='tipo_puntuacion',
        value_name='puntuacion'
    )
    
    # Renombrar las categorías para mejor visualización
    df_melted['tipo_puntuacion'] = df_melted['tipo_puntuacion'].map({
        'critic_score': 'Críticos',
        'audience_score': 'Audiencia'
    })
    
    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Crear el box plot
    sns.boxplot(
        data=df_melted,
        x='genre',
        y='puntuacion',
        hue='tipo_puntuacion',
        palette=['#2ecc71', '#3498db'],  # Verde para críticos, azul para audiencia
        width=0.8,
        ax=ax
    )
    
    # Personalizar el gráfico
    ax.set_title('Distribución de Puntuaciones por Género', pad=20, size=14)
    ax.set_xlabel('Género', size=12)
    ax.set_ylabel('Puntuación', size=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Añadir grid
    ax.grid(True, alpha=0.3)
    
    # Ajustar la leyenda
    ax.legend(
        title='Tipo de Puntuación',
        title_fontsize=10,
        fontsize=10,
        loc='upper right'
    )
    
    # Ajustar el layout
    plt.tight_layout()
    
    return fig

def plot_correlacion_puntuaciones(df):
    """
    Visualiza la correlación entre puntuaciones de críticos y audiencia
    """
    configurar_estilo_plots()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.regplot(data=df, x='critic_score', y='audience_score', ax=ax,
                scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
    
    correlation = df['critic_score'].corr(df['audience_score'])
    
    ax.set_title(f'Correlación entre Puntuaciones\nCorrelación: {correlation:.2f}')
    ax.set_xlabel('Puntuación de Críticos')
    ax.set_ylabel('Puntuación de Audiencia')
    ax.grid(True, alpha=0.3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

def plot_evolucion_temporal_completa(df):
    """
    Crea visualizaciones de la evolución temporal de recaudación y puntuaciones
    incluyendo media, mediana y rango intercuartílico
    """
    configurar_estilo_plots()
    
    # Preparar datos para recaudación
    yearly_boxoffice = df.groupby('release_year')['box_office_millions'].agg([
        ('media', 'mean'),
        ('mediana', 'median'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ]).reset_index()
    
    # Preparar datos para puntuaciones
    yearly_scores = df.groupby('release_year').agg({
        'critic_score': ['mean', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        'audience_score': ['mean', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    }).reset_index()
    
    yearly_scores.columns = ['release_year', 
                           'critic_mean', 'critic_median', 'critic_q25', 'critic_q75',
                           'audience_mean', 'audience_median', 'audience_q25', 'audience_q75']
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # 1. Gráfico de evolución de recaudación
    ax1.plot(yearly_boxoffice['release_year'], yearly_boxoffice['media'], 
             label='Media', color='blue', linewidth=2)
    ax1.plot(yearly_boxoffice['release_year'], yearly_boxoffice['mediana'], 
             label='Mediana', color='red', linewidth=2, linestyle='--')
    ax1.fill_between(yearly_boxoffice['release_year'], 
                     yearly_boxoffice['q25'], 
                     yearly_boxoffice['q75'],
                     alpha=0.2, color='blue', label='Rango Intercuartílico')
    
    ax1.set_title('Evolución Temporal de la Recaudación')
    ax1.set_xlabel('Año de Lanzamiento')
    ax1.set_ylabel('Recaudación (millones)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Gráfico de evolución de puntuaciones
    # Críticos
    ax2.plot(yearly_scores['release_year'], yearly_scores['critic_mean'], 
             label='Media Críticos', color='blue', linewidth=2)
    ax2.plot(yearly_scores['release_year'], yearly_scores['critic_median'], 
             label='Mediana Críticos', color='blue', linewidth=2, linestyle='--')
    ax2.fill_between(yearly_scores['release_year'], 
                     yearly_scores['critic_q25'], 
                     yearly_scores['critic_q75'],
                     alpha=0.2, color='blue', label='IQR Críticos')
    
    # Audiencia
    ax2.plot(yearly_scores['release_year'], yearly_scores['audience_mean'], 
             label='Media Audiencia', color='red', linewidth=2)
    ax2.plot(yearly_scores['release_year'], yearly_scores['audience_median'], 
             label='Mediana Audiencia', color='red', linewidth=2, linestyle='--')
    ax2.fill_between(yearly_scores['release_year'], 
                     yearly_scores['audience_q25'], 
                     yearly_scores['audience_q75'],
                     alpha=0.2, color='red', label='IQR Audiencia')
    
    ax2.set_title('Evolución Temporal de las Puntuaciones')
    ax2.set_xlabel('Año de Lanzamiento')
    ax2.set_ylabel('Puntuación')
    ax2.legend(ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_evolucion_temporal_puntuaciones(df):
    """
    Visualiza la evolución temporal de las puntuaciones
    """
    configurar_estilo_plots()
    
    yearly_scores = df.groupby('release_year').agg({
        'critic_score': 'mean',
        'audience_score': 'mean'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    ax.plot(yearly_scores['release_year'], yearly_scores['critic_score'], 
            label='Críticos', marker='o', linestyle='-', markersize=6)
    ax.plot(yearly_scores['release_year'], yearly_scores['audience_score'], 
            label='Audiencia', marker='s', linestyle='-', markersize=6)
    
    ax.set_title('Evolución Temporal de Puntuaciones Promedio')
    ax.set_xlabel('Año de Lanzamiento')
    ax.set_ylabel('Puntuación Promedio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

def plot_recaudacion_por_genero(df):
    """
    Visualiza la recaudación promedio por género
    """
    configurar_estilo_plots()
    
    # Ordenar los datos por recaudación total de forma descendente
    genre_stats = df.groupby('genre')['box_office_millions'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Usar el orden establecido para el gráfico
    sns.barplot(x=genre_stats.index, y=genre_stats.values, palette='viridis', ax=ax)
    
    ax.set_title('Recaudación Promedio por Género')
    ax.set_xlabel('Género')
    ax.set_ylabel('Recaudación Promedio (millones)')
    ax.tick_params(axis='x', rotation=45)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

# También modificar la función plot_popularidad_por_genero para mantener consistencia:

def plot_popularidad_por_genero(df):
    """
    Visualiza la popularidad promedio por género
    """
    configurar_estilo_plots()
    
    # Ordenar por popularidad de forma descendente
    genre_stats = df.groupby('genre')['popularity'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=genre_stats.index, y=genre_stats.values, palette='magma', ax=ax)
    
    ax.set_title('Popularidad Promedio por Género')
    ax.set_xlabel('Género')
    ax.set_ylabel('Popularidad Promedio')
    ax.tick_params(axis='x', rotation=45)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def generar_informe_visual_completo(df):
    """
    Genera y guarda todas las visualizaciones
    """
    try:
        # 1. Distribuciones generales
        fig_dist = plot_distribucion_puntuaciones(df)
        fig_dist.savefig('distribucion_puntuaciones.png', bbox_inches='tight', dpi=300)
        
        # 2. Distribuciones por género
        fig_genre = plot_distribucion_por_genero(df)
        fig_genre.savefig('distribucion_por_genero.png', bbox_inches='tight', dpi=300)
        
        # 3. Correlación entre puntuaciones
        fig_corr = plot_correlacion_puntuaciones(df)
        fig_corr.savefig('correlacion_puntuaciones.png', bbox_inches='tight', dpi=300)
        
        # 4. Evolución temporal
        fig_temp = plot_evolucion_temporal_puntuaciones(df)
        fig_temp.savefig('evolucion_temporal_puntuaciones.png', bbox_inches='tight', dpi=300)
        
        # 5. Popularidad por género
        fig_pop = plot_popularidad_por_genero(df)
        fig_pop.savefig('popularidad_por_genero.png', bbox_inches='tight', dpi=300)
        
        # 6. Recaudación por género
        fig_box = plot_recaudacion_por_genero(df)
        fig_box.savefig('recaudacion_por_genero.png', bbox_inches='tight', dpi=300)
        
        plt.close('all')
        
        print("Visualizaciones generadas y guardadas exitosamente:")
        print("1. distribucion_puntuaciones.png")
        print("2. distribucion_por_genero.png")
        print("3. correlacion_puntuaciones.png")
        print("4. evolucion_temporal_puntuaciones.png")
        print("5. popularidad_por_genero.png")
        print("6. recaudacion_por_genero.png")
        
    except Exception as e:
        print(f"Error al generar las visualizaciones: {str(e)}")