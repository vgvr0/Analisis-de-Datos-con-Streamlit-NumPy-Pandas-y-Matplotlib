import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats

from analysis import (cargar_y_limpiar_datos, analizar_valores_faltantes,
                     detectar_outliers, calcular_estadisticas_basicas,
                     analizar_correlaciones, analizar_variables_derivadas,
                     realizar_agregaciones, generar_resumen_estadistico, calcular_metricas_adicionales,  generar_analisis_datos_problematicos,
                     crear_sistema_recomendacion, recomendar_peliculas, )

from visualization import (plot_distribucion_puntuaciones, plot_distribucion_por_genero,
                         plot_correlacion_puntuaciones, plot_evolucion_temporal_puntuaciones,
                         plot_popularidad_por_genero, plot_recaudacion_por_genero,
                         plot_evolucion_temporal_completa, plot_distribucion_puntuaciones, plot_correlacion_puntuaciones, plot_evolucion_temporal_puntuaciones, plot_distribucion_por_genero)

def mi_codigo():
    # Configuración de la página
    st.set_page_config(
        page_title="Análisis de Películas",
        page_icon="🎬",
        layout="wide"
    )

    # Título principal
    st.title("🎬 Movies Data Dashboard")
    st.markdown("---")
    
    # Explicación de variables
    with st.expander("📚 Explicación de Variables del Dataset"):
        st.markdown("""
        ### Variables del Dataset
        
        #### Variables Originales
        | Variable | Descripción | Unidad/Formato |
        |----------|-------------|----------------|
        | **title** | Título de la película | Texto |
        | **genre** | Género principal de la película | Categoría |
        | **release_year** | Año de estreno de la película | Año (YYYY) |
        | **critic_score** | Puntuación otorgada por críticos profesionales | 0-100 puntos |
        | **audience_score** | Puntuación otorgada por la audiencia general | 0-100 puntos |
        | **box_office_millions** | Recaudación total en taquilla | Millones de USD ($) |
        | **popularity** | Índice de popularidad basado en interacciones y búsquedas | Escala 0-100 |
        | **duration_minutes** | Duración total de la película | Minutos |
        | **budget_millions** | Presupuesto total de producción | Millones de USD ($) |
        
        #### Variables Derivadas
        | Variable | Descripción | Unidad/Formato |
        |----------|-------------|----------------|
        | **roi** | Retorno de Inversión: ((box_office - budget) / budget) × 100 | Porcentaje (%) |
        | **diferencia_scores** | Diferencia entre puntuación de críticos y audiencia (critic_score - audience_score) | Puntos (-100 a 100) |
        | **categoria_popularidad** | Clasificación de películas según su popularidad | Categorías: Muy Baja, Baja, Media, Alta, Muy Alta |
        
        #### Detalles Adicionales:
        
        - **ROI (Return on Investment)**:
          - Mide la rentabilidad de la película
          - Un ROI de 100% significa que duplicó su inversión
          - ROI negativo indica pérdidas
          - Fórmula: ((box_office - budget) / budget) × 100
        
        - **Diferencia de Puntuaciones**:
          - Valor positivo: Críticos puntuaron más alto que la audiencia
          - Valor negativo: Audiencia puntuó más alto que los críticos
          - Cerca de 0: Consenso entre críticos y audiencia
          - Rango: De -100 (máxima discrepancia a favor de audiencia) a 100 (máxima discrepancia a favor de críticos)
        
        - **Categoría de Popularidad**:
          - **Muy Baja**: 20% inferior del índice de popularidad
          - **Baja**: Entre 20% y 40%
          - **Media**: Entre 40% y 60%
          - **Alta**: Entre 60% y 80%
          - **Muy Alta**: 20% superior del índice de popularidad
        
        - **Puntuaciones (critic_score y audience_score)**:
          - Escala de 0 a 100 donde 100 es la mejor puntuación
          - Las puntuaciones de críticos suelen ser más rigurosas
          - Las puntuaciones de audiencia reflejan la satisfacción general del público
        
        - **Métricas Financieras**:
          - **box_office_millions**: Incluye recaudación global en cines
          - **budget_millions**: Incluye costos de producción y marketing
        
        - **Popularity**:
          - Métrica compuesta que considera:
            - Volumen de búsquedas
            - Interacciones en redes sociales
            - Menciones en medios
            - Tendencias actuales
        
        - **Genre**:
          - Categoría principal de la película
          - Una película solo puede tener un género asignado
          - Basado en la categorización estándar de la industria
        
        - **Duration_minutes**:
          - Tiempo total de reproducción
          - No incluye créditos adicionales o escenas post-créditos
        """)
        
    st.markdown("---")

    # Continuar con el resto del dashboard...

    with st.sidebar:
        st.header("Configuración")
        uploaded_file = st.file_uploader("Cargar dataset de películas", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            # Cargar y procesar dataset
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, sheet_name=0)
            
            df = cargar_y_limpiar_datos(df)
            
            # Contador de películas seleccionadas (movido aquí)
            st.markdown("---")
            current_movies = len(df)  # Se actualizará después de aplicar los filtros
            st.write(f"📽️ **Películas seleccionadas:** {current_movies:,}")
            percentage = 100.0  # Se actualizará después de aplicar los filtros
            st.write(f"📊 **Porcentaje del total:** {percentage:.1f}%")
            st.markdown("---")

            st.subheader("🎯 Filtros")
                        
            # Filtro de géneros mejorado
            st.write("🎭 Géneros")
            if st.button("Seleccionar Todos los Géneros"):
                st.session_state.selected_genres = sorted(df['genre'].unique().tolist())
            
            selected_genres = st.multiselect(
                "",
                options=sorted(df['genre'].unique().tolist()),
                default=st.session_state.get('selected_genres', sorted(df['genre'].unique().tolist()))
            )
            st.session_state.selected_genres = selected_genres
            
            # Filtro de años con input manual
            st.write("📅 Rango de Años")
            col_slider_year, col_inputs_year = st.columns([2, 1])
            
            # Obtener valores min/max para años
            min_year = int(df['release_year'].min())
            max_year = int(df['release_year'].max())
            
            # Valores actuales desde session_state o valores por defecto
            current_min_year = st.session_state.get('year_min', min_year)
            current_max_year = st.session_state.get('year_max', max_year)
            
            with col_slider_year:
                selected_years = st.slider(
                    "",
                    min_value=min_year,
                    max_value=max_year,
                    value=(current_min_year, current_max_year),
                    key="slider_years"
                )
            
            with col_inputs_year:
                input_min_year = st.number_input(
                    "Min",
                    min_value=min_year,
                    max_value=max_year,
                    value=selected_years[0],
                    key="input_min_year"
                )
                input_max_year = st.number_input(
                    "Max",
                    min_value=min_year,
                    max_value=max_year,
                    value=selected_years[1],
                    key="input_max_year"
                )
            
            # Actualizar session_state para años
            st.session_state['year_min'] = input_min_year
            st.session_state['year_max'] = input_max_year
            selected_years = (input_min_year, input_max_year)

            def numeric_filter(label, column, emoji):
                st.write(f"{emoji} {label}")
                
                # Obtener valores min/max del dataset
                min_val = float(df[column].min())
                max_val = float(df[column].max())
                
                # Valores actuales (desde session_state o valores por defecto)
                current_min = st.session_state.get(f'{column}_min', min_val)
                current_max = st.session_state.get(f'{column}_max', max_val)
                
                # Slider para rango de valores
                range_vals = st.slider(
                    "",
                    min_value=min_val,
                    max_value=max_val,
                    value=(current_min, current_max),
                    format="%.1f",
                    key=f"slider_{column}"
                )

                # Inputs manuales debajo del slider
                st.number_input(
                    "Min",
                    min_value=min_val,
                    max_value=max_val,
                    value=range_vals[0],
                    format="%.1f",
                    key=f"input_min_{column}"
                )
                st.number_input(
                    "Max",
                    min_value=min_val,
                    max_value=max_val,
                    value=range_vals[1],
                    format="%.1f",
                    key=f"input_max_{column}"
                )
                
                # Actualizar session_state
                st.session_state[f'{column}_min'] = st.session_state[f"input_min_{column}"]
                st.session_state[f'{column}_max'] = st.session_state[f"input_max_{column}"]
                
                return (st.session_state[f"input_min_{column}"], st.session_state[f"input_max_{column}"])

            
            # Aplicar filtros numéricos
            box_office_range = numeric_filter(
                "Recaudación (Millones $)", 
                'box_office_millions',
                "💰"
            )
            
            audience_score_range = numeric_filter(
                "Puntuación Audiencia",
                'audience_score',
                "👥"
            )
            
            critic_score_range = numeric_filter(
                "Puntuación Críticos",
                'critic_score',
                "🎭"
            )
            
            budget_range = numeric_filter(
                "Presupuesto (Millones $)",
                'budget_millions',
                "💵"
            )
            
            # Aplicar todos los filtros
            df_filtered = df[
                
                (df['release_year'].between(selected_years[0], selected_years[1])) &
                (df['genre'].isin(selected_genres)) &
                (df['box_office_millions'].between(box_office_range[0], box_office_range[1])) &
                (df['audience_score'].between(audience_score_range[0], audience_score_range[1])) &
                (df['critic_score'].between(critic_score_range[0], critic_score_range[1])) &
                (df['budget_millions'].between(budget_range[0], budget_range[1]))
            ]
            
            # Actualizar contador de películas y porcentaje
            current_movies = len(df_filtered)
            percentage = (current_movies / len(df)) * 100
            
        else:
            st.warning("Por favor, carga un archivo CSV o Excel para comenzar el análisis.")
            return

    # Tabs para navegación
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Resumen General",
        "📈 Análisis por Género",
        "💰 Análisis Financiero",
        "📊 Visualizaciones interactivas",
        "📈 Evolución Temporal",
        "🎬 Recomendador"
    ])
# En la sección del tab1 (Resumen General), después de las métricas principales:

    with tab1:
        st.header("Resumen General del Dataset")
        
        # Métricas principales en la parte superior
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Películas", len(df_filtered))
        with col2:
            st.metric("Promedio Críticos", f"{df_filtered['critic_score'].mean():.1f}")
        with col3:
            st.metric("Promedio Audiencia", f"{df_filtered['audience_score'].mean():.1f}")
        with col4:
            st.metric("Box Office Promedio", f"${df_filtered['box_office_millions'].mean():.1f}M")

# En la sección del tab1 (Resumen General), después de las métricas principales:

        # Nueva sección: Indicadores de Éxito
        st.subheader("🎯 Indicadores de Éxito")
        
        # Calcular los indicadores
        total_peliculas = len(df_filtered)
        
        # Éxito ROI
        exito_roi = df_filtered['roi'] >= df_filtered['roi'].median()
        num_exito_roi = exito_roi.sum()
        porc_exito_roi = (num_exito_roi / total_peliculas) * 100
        roi_mediana = df_filtered['roi'].median()
        
        # Éxito Popularidad
        exito_popularidad = df_filtered['popularity'] >= df_filtered['popularity'].median()
        num_exito_popularidad = exito_popularidad.sum()
        porc_exito_popularidad = (num_exito_popularidad / total_peliculas) * 100
        popularidad_mediana = df_filtered['popularity'].median()
        
        # Éxito Audiencia
        exito_audiencia = df_filtered['audience_score'] >= df_filtered['audience_score'].median()
        num_exito_audiencia = exito_audiencia.sum()
        porc_exito_audiencia = (num_exito_audiencia / total_peliculas) * 100
        audiencia_mediana = df_filtered['audience_score'].median()
        
        # Crear tres columnas para mostrar los indicadores
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background-color: #FFF3E0; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #E65100; font-size: 20px;'>💰 Éxito ROI</h3>
                <p style='font-size: 28px; font-weight: bold; color: #E65100;'>{:.1f}%</p>
                <p style='color: #424242;'>{:,} de {:,} películas</p>
                <p style='font-size: 14px; color: #757575;'>Películas con ROI superior a {:.1f}%</p>
            </div>
            """.format(porc_exito_roi, num_exito_roi, total_peliculas, roi_mediana), unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style='background-color: #E8F5E9; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #388E3C; font-size: 20px;'>⭐ Éxito en Popularidad</h3>
                <p style='font-size: 28px; font-weight: bold; color: #388E3C;'>{:.1f}%</p>
                <p style='color: #424242;'>{:,} de {:,} películas</p>
                <p style='font-size: 14px; color: #757575;'>Películas con popularidad superior a {:.1f}</p>
            </div>
            """.format(porc_exito_popularidad, num_exito_popularidad, total_peliculas, popularidad_mediana), unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style='background-color: #F3E5F5; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #7B1FA2; font-size: 20px;'>👥 Éxito con Audiencia</h3>
                <p style='font-size: 28px; font-weight: bold; color: #7B1FA2;'>{:.1f}%</p>
                <p style='color: #424242;'>{:,} de {:,} películas</p>
                <p style='font-size: 14px; color: #757575;'>Películas con puntuación de audiencia superior a {:.1f}</p>
            </div>
            """.format(porc_exito_audiencia, num_exito_audiencia, total_peliculas, audiencia_mediana), unsafe_allow_html=True)
        
        # Análisis de coincidencias y películas exitosas
        st.markdown("### 🔄 Análisis Detallado")
        
        # Crear dos columnas del mismo tamaño
        col1, col2 = st.columns(2)
        
        # Calcular intersección de los tres tipos de éxito
        exito_total = exito_roi & exito_popularidad & exito_audiencia
        num_exito_total = exito_total.sum()
        porc_exito_total = (num_exito_total / total_peliculas) * 100
        
        with col1:
            st.markdown("#### Análisis de Coincidencias")
            st.markdown("""
            <div style='background-color: #E3F2FD; padding: 20px; border-radius: 10px; height: 100%;'>
                <h3 style='color: #1976D2; font-size: 20px;'>🌟 Éxito Total</h3>
                <p style='font-size: 28px; font-weight: bold; color: #1976D2;'>{:.1f}%</p>
                <p style='color: #424242;'>{:,} películas</p>
                <p style='font-size: 14px; color: #757575;'>Películas exitosas en ROI, popularidad y audiencia</p>
            </div>
            """.format(porc_exito_total, num_exito_total), unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Algunas Películas con Éxito Total")
            # Crear un DataFrame con las películas de éxito total
            peliculas_exitosas = df_filtered[exito_total][
                ['title', 'release_year', 'genre', 'roi', 'popularity', 'audience_score', 'box_office_millions']
            ].sort_values('roi', ascending=False).head()
            
            if not peliculas_exitosas.empty:
                st.dataframe(
                    peliculas_exitosas
                    .style.format({
                        'roi': '{:.1f}%',
                        'popularity': '{:.1f}',
                        'audience_score': '{:.1f}',
                        'box_office_millions': '${:,.1f}M'
                    }),
                    height=240
                )
            else:
                st.info("No se encontraron películas con éxito en todas las categorías en el filtro actual")

        # Continuar con el resto del contenido de la pestaña...
        st.markdown("---")

        # Nueva sección: Estadísticas Descriptivas
        st.subheader("📊 Estadísticas Descriptivas")
        stats_df = generar_resumen_estadistico(df_filtered)
        st.dataframe(stats_df, use_container_width=True)
        
        # Sección: Métricas Destacadas
        st.subheader("🏆 Métricas Destacadas")
        metricas = calcular_metricas_adicionales(df_filtered)
        
        # Tabs para diferentes categorías
        tab_rent, tab_budget, tab_critic, tab_aud, tab_pop = st.tabs([
            "💰 Rentabilidad",
            "💵 Presupuesto",
            "🎭 Críticos",
            "👥 Audiencia",
            "⭐ Popularidad"
        ])
        
        with tab_rent:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top 5 Películas más Rentables:**")
                st.dataframe(
                    metricas['Películas más rentables']
                    .rename(columns={
                        'title': 'Título',
                        'release_year': 'Año',
                        'roi': 'ROI (%)',
                        'box_office_millions': 'Box Office (M$)'
                    })
                )
            with col2:
                st.write("**Top 5 Películas menos Rentables:**")
                st.dataframe(
                    metricas['Películas menos rentables']
                    .rename(columns={
                        'title': 'Título',
                        'release_year': 'Año',
                        'roi': 'ROI (%)',
                        'box_office_millions': 'Box Office (M$)'
                    })
                )
        
        with tab_budget:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top 5 Mayor Presupuesto:**")
                st.dataframe(
                    metricas['Mayor presupuesto']
                    .rename(columns={
                        'title': 'Título',
                        'release_year': 'Año',
                        'budget_millions': 'Presupuesto (M$)',
                        'box_office_millions': 'Box Office (M$)'
                    })
                )
            with col2:
                st.write("**Top 5 Menor Presupuesto:**")
                st.dataframe(
                    metricas['Menor presupuesto']
                    .rename(columns={
                        'title': 'Título',
                        'release_year': 'Año',
                        'budget_millions': 'Presupuesto (M$)',
                        'box_office_millions': 'Box Office (M$)'
                    })
                )
        
        with tab_critic:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top 5 Mejor Puntuadas por Críticos:**")
                st.dataframe(
                    metricas['Mejor puntuadas por críticos']
                    .rename(columns={
                        'title': 'Título',
                        'release_year': 'Año',
                        'critic_score': 'Punt. Críticos',
                        'audience_score': 'Punt. Audiencia'
                    })
                )
            with col2:
                st.write("**Top 5 Peor Puntuadas por Críticos:**")
                st.dataframe(
                    metricas['Peor puntuadas por críticos']
                    .rename(columns={
                        'title': 'Título',
                        'release_year': 'Año',
                        'critic_score': 'Punt. Críticos',
                        'audience_score': 'Punt. Audiencia'
                    })
                )
        
        with tab_aud:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top 5 Mejor Puntuadas por Audiencia:**")
                st.dataframe(
                    metricas['Mejor puntuadas por audiencia']
                    .rename(columns={
                        'title': 'Título',
                        'release_year': 'Año',
                        'audience_score': 'Punt. Audiencia',
                        'critic_score': 'Punt. Críticos'
                    })
                )
            with col2:
                st.write("**Top 5 Peor Puntuadas por Audiencia:**")
                st.dataframe(
                    metricas['Peor puntuadas por audiencia']
                    .rename(columns={
                        'title': 'Título',
                        'release_year': 'Año',
                        'audience_score': 'Punt. Audiencia',
                        'critic_score': 'Punt. Críticos'
                    })
                )
        
        with tab_pop:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top 5 Películas más Populares:**")
                st.dataframe(
                    metricas['Películas más populares']
                    .rename(columns={
                        'title': 'Título',
                        'release_year': 'Año',
                        'popularity': 'Popularidad',
                        'box_office_millions': 'Box Office (M$)'
                    })
                )
            with col2:
                st.write("**Top 5 Películas menos Populares:**")
                st.dataframe(
                    metricas['Películas menos populares']
                    .rename(columns={
                        'title': 'Título',
                        'release_year': 'Año',
                        'popularity': 'Popularidad',
                        'box_office_millions': 'Box Office (M$)'
                    })
                )
        

    # Distribución por género y estadísticas
        st.subheader("🎭 Distribución y Estadísticas por Género")
        
        # Crear tres columnas
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Gráfico de distribución
            fig_genre = px.pie(
                df_filtered, 
                names='genre', 
                title='Distribución de Películas por Género',
                hole=0.4  # Hacer un donut chart para mejor visualización
            )
            # Actualizar layout para mejor visualización
            fig_genre.update_traces(
                textposition='inside',
                textinfo='percent+label'
            )
            fig_genre.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig_genre, use_container_width=True)
        
        with col2:
            st.write("**Géneros más Comunes:**")
            st.dataframe(
                metricas['Géneros más comunes']
                .rename("Cantidad de Películas"),
                height=300
            )
        
        with col3:
            st.write("**Duración Promedio:**")
            st.dataframe(
                metricas['Duración promedio por género']
                .rename("Minutos"),
                height=300
            )

# Timeline de películas mejorado
        st.subheader("📅 Timeline de Películas")

        # Preparar datos para el gráfico
        timeline_data = df_filtered.groupby('release_year').size().reset_index(name='count')
        timeline_data['decade'] = (timeline_data['release_year'] // 10) * 10

        # Crear el gráfico
        fig_timeline = go.Figure()

        # Añadir el área base
        fig_timeline.add_trace(go.Scatter(
            x=timeline_data['release_year'],
            y=timeline_data['count'],
            fill='tozeroy',
            name='Películas por Año',
            line=dict(width=2, color='#1f77b4'),
            fillcolor='rgba(31, 119, 180, 0.3)',
            hovertemplate='Año: %{x}<br>Películas: %{y}<extra></extra>'
        ))

        # Añadir puntos para cada año
        fig_timeline.add_trace(go.Scatter(
            x=timeline_data['release_year'],
            y=timeline_data['count'],
            mode='markers',
            name='Cantidad',
            marker=dict(
                size=8,
                color='#1f77b4',
                line=dict(
                    color='white',
                    width=1
                )
            ),
            hovertemplate='Año: %{x}<br>Películas: %{y}<extra></extra>'
        ))

        # Personalizar el layout
        fig_timeline.update_layout(
            title={
                'text': 'Evolución Temporal de Estrenos',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Año de Estreno",
            yaxis_title="Número de Películas",
            hovermode='x unified',
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Añadir grid y personalizar ejes
        fig_timeline.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            dtick=5,  # Marcas cada 5 años
            tickangle=45
        )

        fig_timeline.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(128, 128, 128, 0.2)'
        )

        # Añadir anotaciones para décadas significativas
        decades = timeline_data.groupby('decade')['count'].sum().reset_index()
        for _, decade in decades.iterrows():
            if decade['count'] > 0:  # Solo añadir anotación si hay películas
                fig_timeline.add_annotation(
                    x=decade['decade'],
                    y=timeline_data[timeline_data['release_year'] == decade['decade']]['count'].iloc[0],
                    text=f"Década {decade['decade']}s",
                    showarrow=False,
                    yshift=20,
                    font=dict(size=10)
                )

        # Mostrar estadísticas debajo del gráfico
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Añadir métricas resumidas
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            año_mas_peliculas = timeline_data.loc[timeline_data['count'].idxmax()]
            st.metric(
                "Año más Productivo",
                f"{int(año_mas_peliculas['release_year'])}",
                f"{int(año_mas_peliculas['count'])} películas"
            )

        with col2:
            decada_mas_peliculas = decades.loc[decades['count'].idxmax()]
            st.metric(
                "Década más Productiva",
                f"{int(decada_mas_peliculas['decade'])}s",
                f"{int(decada_mas_peliculas['count'])} películas"
            )

        with col3:
            promedio_anual = timeline_data['count'].mean()
            st.metric(
                "Promedio Anual",
                f"{promedio_anual:.1f}",
                "películas/año"
            )

        with col4:
            total_years = timeline_data['release_year'].max() - timeline_data['release_year'].min()
            st.metric(
                "Período Analizado",
                f"{total_years} años",
                f"{len(df_filtered)} películas totales"
            )
            
        # Nueva sección de boxplots
        st.subheader("📊 Distribución de Variables Principales")
        
        # Lista de variables para analizar
        variables_numericas = {
            'critic_score': 'Puntuación de Críticos',
            'audience_score': 'Puntuación de Audiencia',
            'box_office_millions': 'Recaudación (M$)',
            'budget_millions': 'Presupuesto (M$)',
            'popularity': 'Popularidad',
            'duration_minutes': 'Duración (min)',
            'roi': 'ROI (%)',
            'diferencia_scores': 'Diferencia Críticos-Audiencia'
        }
        
        # Crear figura con subplots
        fig = make_subplots(
            rows=2, 
            cols=4,
            subplot_titles=list(variables_numericas.values())
        )
        
        # Colores para los boxplots
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', 
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # Añadir boxplots
        row = 1
        col = 1
        for i, (variable, titulo) in enumerate(variables_numericas.items()):
            fig.add_trace(
                go.Box(
                    y=df_filtered[variable],
                    name=titulo,
                    boxpoints='outliers',
                    marker_color=colors[i],
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{titulo}</b><br>" +
                        "Valor: %{y}<br>" +
                        "<extra></extra>"
                    )
                ),
                row=row, 
                col=col
            )
            
            col += 1
            if col > 4:
                col = 1
                row += 1
        
        # Actualizar layout
        fig.update_layout(
            height=800,
            width=1200,
            showlegend=False,
            title={
                'text': 'Distribución y Outliers de Variables Principales',
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            font=dict(size=10),
            boxmode='group',
            template='plotly_white'
        )
        
        # Ajustar márgenes y espaciado
        fig.update_layout(
            margin=dict(l=50, r=50, t=100, b=50),
            boxgap=0.3
        )
        
        # Añadir tooltips informativos
        tooltips = {
            'critic_score': 'Puntuación de 0 a 100 otorgada por críticos profesionales',
            'audience_score': 'Puntuación de 0 a 100 otorgada por la audiencia',
            'box_office_millions': 'Recaudación total en millones de dólares',
            'budget_millions': 'Presupuesto total en millones de dólares',
            'popularity': 'Índice de popularidad (0-100)',
            'duration_minutes': 'Duración de la película en minutos',
            'roi': 'Retorno de Inversión en porcentaje',
            'diferencia_scores': 'Diferencia entre puntuación de críticos y audiencia'
        }
        
        # Actualizar títulos de ejes y formato
        for i, (variable, titulo) in enumerate(variables_numericas.items()):
            row = (i // 4) + 1
            col = (i % 4) + 1
            
            # Actualizar formato de ejes
            if variable in ['box_office_millions', 'budget_millions']:
                fig.update_yaxes(
                    title_text='Millones $',
                    row=row, 
                    col=col,
                    tickformat='$,.0f'
                )
            elif variable in ['roi']:
                fig.update_yaxes(
                    title_text='%',
                    row=row, 
                    col=col,
                    tickformat='.1f'
                )
            elif variable in ['duration_minutes']:
                fig.update_yaxes(
                    title_text='Minutos',
                    row=row, 
                    col=col
                )
            else:
                fig.update_yaxes(
                    title_text='Valor',
                    row=row, 
                    col=col
                )
        
        # Mostrar el gráfico
        st.plotly_chart(fig, use_container_width=True)
        
        # Añadir explicación
        with st.expander("ℹ️ Interpretación de los Boxplots"):
            st.markdown("""
            ### Cómo interpretar los Boxplots:
            
            - **Caja central**: Representa el rango intercuartílico (IQR) que contiene el 50% central de los datos
            - Límite superior de la caja: 75° percentil (Q3)
            - Línea central: Mediana
            - Límite inferior de la caja: 25° percentil (Q1)
            
            - **Bigotes**: Extienden hasta 1.5 × IQR desde los bordes de la caja
            - Bigote superior: Hasta el valor máximo dentro de 1.5 × IQR desde Q3
            - Bigote inferior: Hasta el valor mínimo dentro de 1.5 × IQR desde Q1
            
            - **Puntos individuales**: Representan outliers (valores atípicos)
            - Valores por encima del bigote superior o por debajo del bigote inferior
            
            ### Interpretación por Variable:
            
            - **Puntuaciones (Críticos y Audiencia)**: Distribución de calificaciones en escala 0-100
            - **Recaudación y Presupuesto**: Distribución de valores monetarios, outliers suelen ser blockbusters
            - **Popularidad**: Distribución del índice de popularidad
            - **Duración**: Distribución de tiempos de película
            - **ROI**: Distribución del retorno de inversión, outliers son éxitos o fracasos notables
            - **Diferencia Críticos-Audiencia**: Muestra el desacuerdo entre críticos y audiencia
            """)

    # En la sección de análisis de outliers:

        st.subheader("📊 Análisis de Calidad de Datos")
        
        # Generar el análisis incluyendo diferencia_scores
        columnas_numericas = [
            'critic_score', 'audience_score', 'box_office_millions',
            'popularity', 'duration_minutes', 'budget_millions', 'roi',
            'diferencia_scores'  # Añadida nueva variable
        ]
        
        # Calcular outliers
        outliers_info = {}
        
        for columna in columnas_numericas:
            Q1 = df_filtered[columna].quantile(0.25)
            Q3 = df_filtered[columna].quantile(0.75)
            IQR = Q3 - Q1
            
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            
            outliers = df_filtered[(df_filtered[columna] < limite_inferior) | 
                                (df_filtered[columna] > limite_superior)][columna]
            
            outliers_info[columna] = {
                'cantidad': len(outliers),
                'porcentaje': (len(outliers) / len(df_filtered)) * 100,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'limite_inferior': limite_inferior,
                'limite_superior': limite_superior
            }
        
        # Crear DataFrame para mostrar resultados
        analisis_df = pd.DataFrame({
            'Valores Faltantes': df_filtered[columnas_numericas].isnull().sum(),
            'Porcentaje Faltantes (%)': (df_filtered[columnas_numericas].isnull().sum() / len(df_filtered) * 100).round(2),
            'Cantidad Outliers': [outliers_info[col]['cantidad'] for col in columnas_numericas],
            'Porcentaje Outliers (%)': [outliers_info[col]['porcentaje'] for col in columnas_numericas]
        })
        
        # Mostrar tablas de análisis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📉 Análisis de Valores Faltantes:**")
            missing_df = analisis_df[['Valores Faltantes', 'Porcentaje Faltantes (%)']].copy()
            missing_df = missing_df[missing_df['Valores Faltantes'] > 0]
            if len(missing_df) > 0:
                st.dataframe(
                    missing_df.style.format({
                        'Valores Faltantes': '{:,.0f}',
                        'Porcentaje Faltantes (%)': '{:.2f}%'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No hay valores faltantes en el dataset 🎉")
        
        with col2:
            st.write("**📊 Análisis de Outliers:**")
            outliers_df = analisis_df[['Cantidad Outliers', 'Porcentaje Outliers (%)']].copy()
            outliers_df = outliers_df[outliers_df['Cantidad Outliers'] > 0]
            if len(outliers_df) > 0:
                st.dataframe(
                    outliers_df.style.format({
                        'Cantidad Outliers': '{:,.0f}',
                        'Porcentaje Outliers (%)': '{:.2f}%'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No se detectaron outliers en el dataset 🎉")
        
        # Si hay outliers, mostrar visualizaciones detalladas
        variables_con_outliers = outliers_df.index.tolist()
        if variables_con_outliers:
            st.write("---")
            st.write("**📈 Análisis Detallado de Variables con Outliers:**")
            
            for variable in variables_con_outliers:
                st.write(f"### Análisis de '{variable}'")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Boxplot
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=df_filtered[variable],
                        name=variable,
                        boxpoints='outliers',
                        jitter=0.3,
                        pointpos=-1.8,
                        marker_color='rgb(7,40,89)'
                    ))
                    
                    # Añadir título específico para diferencia_scores
                    title = variable
                    if variable == 'diferencia_scores':
                        title = 'Diferencia entre Puntuaciones (Críticos - Audiencia)'
                    
                    fig_box.update_layout(
                        title=f'Boxplot con Outliers - {title}',
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                with col2:
                    # Histograma con distribución normal
                    fig_hist = go.Figure()
                    
                    # Añadir histograma
                    fig_hist.add_trace(go.Histogram(
                        x=df_filtered[variable],
                        name='Frecuencia',
                        nbinsx=30,
                        marker_color='rgb(7,40,89)'
                    ))
                    
                    # Calcular y añadir curva de distribución normal
                    x_range = np.linspace(
                        df_filtered[variable].min(),
                        df_filtered[variable].max(),
                        100
                    )
                    mean = df_filtered[variable].mean()
                    std = df_filtered[variable].std()
                    y_norm = stats.norm.pdf(x_range, mean, std)
                    y_norm_scaled = y_norm * len(df_filtered[variable]) * (
                        df_filtered[variable].max() - df_filtered[variable].min()
                    ) / 30
                    
                    fig_hist.add_trace(go.Scatter(
                        x=x_range,
                        y=y_norm_scaled,
                        name='Distribución Normal',
                        line=dict(color='red')
                    ))
                    
                    # Añadir título específico para diferencia_scores
                    if variable == 'diferencia_scores':
                        title = 'Distribución de Diferencias entre Puntuaciones'
                    else:
                        title = f'Distribución de {variable}'
                    
                    fig_hist.update_layout(
                        title=title,
                        xaxis_title=variable,
                        yaxis_title='Frecuencia',
                        height=400
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Estadísticas descriptivas
                stats_df = pd.DataFrame({
                    'Estadística': [
                        'Media',
                        'Mediana',
                        'Desv. Estándar',
                        'Mínimo',
                        'Q1',
                        'Q3',
                        'Máximo',
                        'IQR',
                        'Límite Inferior',
                        'Límite Superior'
                    ],
                    'Valor': [
                        df_filtered[variable].mean(),
                        df_filtered[variable].median(),
                        df_filtered[variable].std(),
                        df_filtered[variable].min(),
                        outliers_info[variable]['Q1'],
                        outliers_info[variable]['Q3'],
                        df_filtered[variable].max(),
                        outliers_info[variable]['IQR'],
                        outliers_info[variable]['limite_inferior'],
                        outliers_info[variable]['limite_superior']
                    ]
                })
                
                st.write("**Estadísticas Descriptivas:**")
                st.dataframe(
                    stats_df.style.format({
                        'Valor': '{:,.2f}'
                    }),
                    use_container_width=True
                )
                
                # Para diferencia_scores, agregar interpretación especial
                if variable == 'diferencia_scores':
                    st.markdown("""
                    **Interpretación de la Diferencia de Puntuaciones:**
                    - Valores **positivos**: Los críticos puntuaron más alto que la audiencia
                    - Valores **negativos**: La audiencia puntuó más alto que los críticos
                    - Valores **cercanos a 0**: Consenso entre críticos y audiencia
                    - **Outliers positivos**: Películas con valoración significativamente mayor por parte de los críticos
                    - **Outliers negativos**: Películas con valoración significativamente mayor por parte de la audiencia
                    """)
                
                st.write("---")


    with tab2:
        st.header("Análisis por Género")
        
        # Dividir en columnas para mejor organización
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Popularidad por Género")
            fig_popularidad = plot_popularidad_por_genero(df_filtered)
            st.pyplot(fig_popularidad)
            
            # Gráfico de tendencias temporales por género
            st.subheader("Tendencias Temporales por Género")
            fig_trends = px.line(
                df_filtered.groupby(['release_year', 'genre'])['popularity'].mean().reset_index(),
                x='release_year',
                y='popularity',
                color='genre',
                title='Evolución de Popularidad por Género'
            )
            st.plotly_chart(fig_trends)
            
        with col2:
            st.subheader("Recaudación por Género")
            fig_recaudacion = plot_recaudacion_por_genero(df_filtered)
            st.pyplot(fig_recaudacion)
            
            st.subheader("ROI Promedio por Género")
            
            # Calcular ROI promedio por género
            df_filtered['roi'] = ((df_filtered['box_office_millions'] - df_filtered['budget_millions']) / 
                                df_filtered['budget_millions']) * 100
            roi_por_genero = df_filtered.groupby('genre')['roi'].mean().reset_index()
            
            # Crear gráfico de barras para ROI por género usando Matplotlib
            fig_roi_genero, ax_roi_genero = plt.subplots(figsize=(8, 6))
            
            # Definir colores para las barras
            colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            
            # Graficar las barras con los colores definidos
            ax_roi_genero.bar(roi_por_genero['genre'], roi_por_genero['roi'], color=colores)
            
            # Configurar las etiquetas y título del gráfico
            ax_roi_genero.set_xlabel('Género')
            ax_roi_genero.set_ylabel('ROI Promedio (%)')
            ax_roi_genero.set_title('ROI Promedio por Género')
            
            # Ajustar las etiquetas del eje x para que no se solapen
            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.25)
            
            st.pyplot(fig_roi_genero)


        # Distribución de puntuaciones
        st.subheader("Distribución de Puntuaciones")
        fig = plot_distribucion_puntuaciones(df_filtered)
        st.pyplot(fig)

        # Correlación entre puntuaciones  
        st.subheader("Correlación entre Puntuaciones")
        fig_corr = plot_correlacion_puntuaciones(df_filtered)
        st.pyplot(fig_corr)

        # Evolución temporal
        st.subheader("Evolución Temporal de Puntuaciones")  
        fig_temp = plot_evolucion_temporal_puntuaciones(df_filtered)
        st.pyplot(fig_temp)

        # Puntuaciones por género
        st.subheader("Puntuaciones por Género")
        fig_genre = plot_distribucion_por_genero(df_filtered)  
        st.pyplot(fig_genre)

    with tab3:
        st.header("Análisis Financiero")
        
        # Métricas financieras
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Box Office Total", f"${df_filtered['box_office_millions'].sum():.1f}M")
        with col2:
            st.metric("Presupuesto Total", f"${df_filtered['budget_millions'].sum():.1f}M")
        with col3:
            roi_promedio = ((df_filtered['box_office_millions'] - df_filtered['budget_millions']) / 
                        df_filtered['budget_millions']).mean() * 100
            st.metric("ROI Promedio", f"{roi_promedio:.1f}%")

        # Crear columnas para mostrar las tablas lado a lado
        col1, col2 = st.columns(2)

        with col1:
            # Top películas por box office
            st.subheader("Top 10 Películas Recaudación")
            top_box_office = df_filtered.nlargest(10, 'box_office_millions')[
                ['title', 'release_year', 'box_office_millions', 'genre']
            ]
            st.dataframe(top_box_office)

        with col2:
            # Cálculo del ROI para cada película
            df_filtered['roi'] = ((df_filtered['box_office_millions'] - df_filtered['budget_millions']) / 
                                df_filtered['budget_millions']) * 100

            # Seleccionar las 10 películas con mejor ROI
            top_roi_movies = df_filtered.nlargest(10, 'roi')[['title', 'release_year', 'box_office_millions', 'budget_millions', 'roi', 'genre']]

            # Mostrar la tabla de las Top 10 películas con mejor ROI
            st.subheader("Top 10 Películas con Mejor ROI")
            st.dataframe(top_roi_movies)

        # Gráfico de barras para visualizar el ROI de las Top 10 películas
        st.subheader("ROI de las Top 10 Películas con Mejor ROI")
        fig_roi_top10 = px.bar(
            top_roi_movies.sort_values(by='roi', ascending=True),  # Orden descendente para gráfico horizontal
            x='roi',
            y='title',
            orientation='h',
            color='genre',
            title="Top 10 Películas con Mejor ROI",
            labels={'roi': 'ROI (%)', 'title': 'Título'},
            color_discrete_sequence=px.colors.qualitative.Pastel  # Escala de colores suave
        )
        st.plotly_chart(fig_roi_top10, use_container_width=True)


        # Análisis de los géneros en las Top 10 películas con mejor ROI
        st.subheader("Resumen de Géneros en las Top 10 Películas con Mejor ROI")
        genre_counts = top_roi_movies['genre'].value_counts()
        st.write("Distribución de géneros entre las 10 películas más rentables en términos de ROI:")
        for genre, count in genre_counts.items():
            st.write(f"- **{genre}**: {count} películas")

        # Análisis de la relación entre presupuesto y ROI en las películas de alto ROI
        st.subheader("Análisis de Presupuesto en las Películas con Mejor ROI")
        average_budget_top_roi = top_roi_movies['budget_millions'].mean()
        st.write(f"El **presupuesto promedio** de las 10 películas con mayor ROI es de **${average_budget_top_roi:.1f}M**.")
        st.write("Esto puede indicar que algunas de las películas más rentables en términos de ROI no necesariamente tuvieron los mayores presupuestos.")


        # # Gráfico de scatter box office vs budget
        # st.subheader("Box Office vs Budget")
        # fig_scatter = px.scatter(
        #     df_filtered,
        #     x='budget_millions',
        #     y='box_office_millions',
        #     color='genre',
        #     size='popularity',
        #     hover_data=['title', 'release_year'],
        #     title='Box Office vs Budget por Género'
        # )
        # st.plotly_chart(fig_scatter)

        # Nueva sección: Análisis de Recaudaciones por Género
        st.subheader("📊 Análisis de Recaudaciones por Género")
        
        # Calcular métricas por género
        generos_stats = df_filtered.groupby('genre').agg({
            'box_office_millions': ['mean', 'median', 'sum', 'count', 'std'],
            'budget_millions': 'mean',
            'roi': 'mean'
        }).round(2)

        # Reorganizar y renombrar columnas
        generos_stats.columns = [
            'Recaudación Media (M$)', 
            'Recaudación Mediana (M$)', 
            'Recaudación Total (M$)',
            'Número de Películas',
            'Desviación Estándar (M$)',
            'Presupuesto Medio (M$)',
            'ROI Medio (%)'
        ]
        generos_stats = generos_stats.sort_values('Recaudación Total (M$)', ascending=False)

        # Crear visualizaciones
        col1, col2 = st.columns(2)

        with col1:
            # Ordenar los datos por ROI Medio de forma descendente
            generos_stats_sorted = generos_stats.sort_values(by='ROI Medio (%)', ascending=False).reset_index()

            # Gráfico de barras de recaudación media por género, ordenado por ROI Medio descendente
            st.subheader("Recaudación Media por Género con Escala de ROI")
            fig_mean = px.bar(
                generos_stats_sorted,
                x='genre',
                y='Recaudación Media (M$)',
                title='Recaudación Media por Género (Escala de Color según ROI Medio)',
                color='ROI Medio (%)',
                labels={'genre': 'Género', 'Recaudación Media (M$)': 'Recaudación Media (M$)'},
                color_continuous_scale='RdYlBu'
            )
            fig_mean.update_layout(showlegend=False)
            st.plotly_chart(fig_mean, use_container_width=True)

        with col2:
            # Gráfico de barras de recaudación total por género
            st.subheader("Recaudación Total por Género")
            fig_total = px.bar(
                generos_stats_sorted,
                x='genre',
                y='Recaudación Total (M$)',
                title='Recaudación Total por Género',
                color='Recaudación Total (M$)',
                labels={'genre': 'Género', 'Recaudación Total (M$)': 'Recaudación Total (M$)'},
                color_continuous_scale='Viridis'
            )
            fig_total.update_layout(showlegend=False)
            st.plotly_chart(fig_total, use_container_width=True)

        # Box plot de distribución de recaudaciones por género
        st.subheader("Distribución de Recaudaciones por Género")
        fig_box = px.box(
            df_filtered,
            x='genre',
            y='box_office_millions',
            title='Distribución de Recaudaciones por Género',
            labels={
                'genre': 'Género',
                'box_office_millions': 'Recaudación (M$)'
            },
            color='genre'
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)


        # Tabla de estadísticas
        with st.expander("📊 Ver Estadísticas Detalladas por Género"):
            st.markdown("### Métricas Detalladas por Género")
            st.dataframe(
                generos_stats.style.format({
                    'Recaudación Media (M$)': '${:,.2f}M',
                    'Recaudación Mediana (M$)': '${:,.2f}M',
                    'Recaudación Total (M$)': '${:,.2f}M',
                    'Número de Películas': '{:,.0f}',
                    'Desviación Estándar (M$)': '${:,.2f}M',
                    'Presupuesto Medio (M$)': '${:,.2f}M',
                    'ROI Medio (%)': '{:,.2f}%'
                }).background_gradient(subset=['Recaudación Total (M$)', 'Recaudación Media (M$)', 'ROI Medio (%)'], cmap='YlOrRd'),
                use_container_width=True
            )

            # Análisis adicional de rentabilidad
            st.markdown("### 💰 Análisis de Rentabilidad por Género")
            
            # Calcular métricas de rentabilidad
            rentabilidad = pd.DataFrame({
                'Género': generos_stats.index,
                'ROI Medio (%)': generos_stats['ROI Medio (%)'],
                'Recaudación/Presupuesto': (generos_stats['Recaudación Media (M$)'] / generos_stats['Presupuesto Medio (M$)']).round(2),
                'Beneficio Medio (M$)': (generos_stats['Recaudación Media (M$)'] - generos_stats['Presupuesto Medio (M$)']).round(2)
            }).sort_values('ROI Medio (%)', ascending=False)

            # Mostrar tabla de rentabilidad
            st.dataframe(
                rentabilidad.style.format({
                    'ROI Medio (%)': '{:,.2f}%',
                    'Recaudación/Presupuesto': '{:,.2f}x',
                    'Beneficio Medio (M$)': '${:,.2f}M'
                }).background_gradient(cmap='RdYlGn'),
                use_container_width=True
            )

            # Insights principales
            st.markdown("""
            ### 🔍 Insights Principales
            
            1. **Género más Taquillero:**
            - Por recaudación total: {}
            - Por recaudación media: {}
            
            2. **Género más Rentable:**
            - Mayor ROI: {} ({:.1f}%)
            - Mayor ratio recaudación/presupuesto: {} ({:.1f}x)
            
            3. **Consistencia:**
            - Género más consistente (menor desviación estándar): {}
            - Género más variable (mayor desviación estándar): {}
            
            4. **Volumen:**
            - Género con más películas: {} ({} películas)
            - Género con menos películas: {} ({} películas)
            """.format(
                generos_stats.index[0],
                generos_stats.sort_values('Recaudación Media (M$)', ascending=False).index[0],
                rentabilidad.iloc[0]['Género'], rentabilidad.iloc[0]['ROI Medio (%)'],
                rentabilidad.sort_values('Recaudación/Presupuesto', ascending=False).iloc[0]['Género'],
                rentabilidad.sort_values('Recaudación/Presupuesto', ascending=False).iloc[0]['Recaudación/Presupuesto'],
                generos_stats.sort_values('Desviación Estándar (M$)').index[0],
                generos_stats.sort_values('Desviación Estándar (M$)', ascending=False).index[0],
                generos_stats.sort_values('Número de Películas', ascending=False).index[0],
                int(generos_stats['Número de Películas'].max()),
                generos_stats.sort_values('Número de Películas').index[0],
                int(generos_stats['Número de Películas'].min())
            ))


    with tab4:
        st.header("Visualizaciones interactivas")
        
        # st.subheader("Evolución Temporal Detallada")
        # fig_temporal = plot_evolucion_temporal_completa(df_filtered)
        # st.pyplot(fig_temporal)

        st.subheader("📊 Análisis por Métrica")
        metrica_seleccionada = st.selectbox(
            "Selecciona la métrica a analizar:",
            [
                "Recaudación (Box Office)",
                "Popularidad",
                "Presupuesto",
                "ROI",
                "Puntuación Críticos",
                "Puntuación Audiencia"
            ]
        )

        # Mapeo de métricas a columnas del DataFrame
        mapeo_metricas = {
            "Recaudación (Box Office)": "box_office_millions",
            "Popularidad": "popularity",
            "Presupuesto": "budget_millions",
            "ROI": "roi",
            "Puntuación Críticos": "critic_score",
            "Puntuación Audiencia": "audience_score"
        }

        # Mapeo de unidades para las etiquetas
        mapeo_unidades = {
            "Recaudación (Box Office)": "(M$)",
            "Popularidad": "",
            "Presupuesto": "(M$)",
            "ROI": "(%)",
            "Puntuación Críticos": "",
            "Puntuación Audiencia": ""
        }

        columna_metrica = mapeo_metricas[metrica_seleccionada]
        unidad = mapeo_unidades[metrica_seleccionada]

        # Calcular estadísticas
        datos_ordenados = df_filtered.groupby('genre').agg({
            columna_metrica: ['sum', 'mean', lambda x: x.quantile(0.25), 'median', lambda x: x.quantile(0.75)]
        }).round(2)

        datos_ordenados.columns = ['total', 'media', 'q25', 'mediana', 'q75']

        # Selector para tipo de ordenamiento
        criterio_orden = st.radio(
            "Ordenar por:",
            ["Total", "Promedio"],
            horizontal=True
        )

        datos_ordenados = datos_ordenados.sort_values(
            'total' if criterio_orden == "Total" else 'media', 
            ascending=False
        )

        # Crear los gráficos
        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de valores totales
            fig_total = px.bar(
                datos_ordenados.reset_index(),
                x='genre',
                y='total',
                title=f'{metrica_seleccionada} Total por Género {unidad}',
                color='total',
                labels={'genre': 'Género', 'total': f'{metrica_seleccionada} Total {unidad}'},
                color_continuous_scale='Viridis'
            )
            fig_total.update_layout(showlegend=False)
            st.plotly_chart(fig_total, use_container_width=True)

        with col2:
            # Gráfico de valores promedio
            fig_mean = px.bar(
                datos_ordenados.reset_index(),
                x='genre',
                y='media',
                title=f'{metrica_seleccionada} Promedio por Género {unidad}',
                color='media',
                labels={'genre': 'Género', 'media': f'{metrica_seleccionada} Promedio {unidad}'},
                color_continuous_scale='Viridis'
            )
            fig_mean.update_layout(showlegend=False)
            st.plotly_chart(fig_mean, use_container_width=True)

        # Box plot para distribución
        fig_box = px.box(
            df_filtered,
            x='genre',
            y=columna_metrica,
            category_orders={'genre': datos_ordenados.index.tolist()},
            title=f'Distribución de {metrica_seleccionada} por Género {unidad}',
            labels={
                'genre': 'Género',
                columna_metrica: f'{metrica_seleccionada} {unidad}'
            }
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Mostrar estadísticas detalladas
        with st.expander("📊 Ver Estadísticas Detalladas"):
            st.dataframe(
                datos_ordenados.style.format({
                    'total': lambda x: f'{x:,.2f}',
                    'media': lambda x: f'{x:,.2f}',
                    'q25': lambda x: f'{x:,.2f}',
                    'mediana': lambda x: f'{x:,.2f}',
                    'q75': lambda x: f'{x:,.2f}',
                }).background_gradient(subset=['total', 'media'], cmap='YlOrRd'),
                use_container_width=True
            )
            datos_ordenados['iqr'] = datos_ordenados['q75'] - datos_ordenados['q25']
            datos_ordenados_por_iqr = datos_ordenados.sort_values('iqr', ascending=False)

            # Calcular y mostrar insights
            st.markdown(f"""
                ### 🔍 Insights Principales para {metrica_seleccionada}
                
                1. **Género con mayor {metrica_seleccionada}:**
                - Por total: {datos_ordenados.index[0]} ({datos_ordenados['total'].iloc[0]:,.2f} {unidad})
                - Por promedio: {datos_ordenados.sort_values('media', ascending=False).index[0]} ({datos_ordenados['media'].max():,.2f} {unidad})
                
                2. **Género con menor {metrica_seleccionada}:**
                - Por total: {datos_ordenados.index[-1]} ({datos_ordenados['total'].iloc[-1]:,.2f} {unidad})
                - Por promedio: {datos_ordenados.sort_values('media', ascending=True).index[0]} ({datos_ordenados['media'].min():,.2f} {unidad})
                
                3. **Variabilidad:**
                - Mayor rango intercuartil: {datos_ordenados_por_iqr.index[0]} (IQR: {datos_ordenados_por_iqr['iqr'].iloc[0]:,.2f} {unidad})
                - Menor rango intercuartil: {datos_ordenados_por_iqr.index[-1]} (IQR: {datos_ordenados_por_iqr['iqr'].iloc[-1]:,.2f} {unidad})
                """)
            
        # Segundo bloque de Análisis Interactivo (análisis detallado)
        st.subheader("Análisis Interactivo Detallado")
        metric_x_detailed = st.selectbox(
            "Seleccionar métrica para eje X",
            ['critic_score', 'audience_score', 'popularity', 'box_office_millions', 'budget_millions'],
            key="metric_x_detailed"  # Key única para el segundo bloque
        )

        metric_y_detailed = st.selectbox(
            "Seleccionar métrica para eje Y",
            ['box_office_millions', 'popularity', 'critic_score', 'audience_score', 'budget_millions'],
            key="metric_y_detailed"  # Key única para el segundo bloque
        )

        # Crear dos columnas para mostrar ambos gráficos
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Análisis por Película**")
            fig_custom_detailed = px.scatter(
                df_filtered,
                x=metric_x_detailed,
                y=metric_y_detailed,
                color='genre',
                size='popularity',
                hover_data=['title', 'release_year'],
                title=f'Relación entre {metric_x_detailed} y {metric_y_detailed} por Película'
            )
            st.plotly_chart(fig_custom_detailed, use_container_width=True)

        with col2:
            st.write("**Análisis por Género**")
            # Calcular promedios por género
            df_grouped = df_filtered.groupby('genre').agg({
                metric_x_detailed: 'mean',
                metric_y_detailed: 'mean',
                'popularity': 'mean'
            }).reset_index()
            
            # Crear scatter plot con los datos agrupados
            fig_grouped = px.scatter(
                df_grouped,
                x=metric_x_detailed,
                y=metric_y_detailed,
                color='genre',
                size='popularity',
                text='genre',
                title=f'Relación entre {metric_x_detailed} y {metric_y_detailed} por Género',
                labels={
                    metric_x_detailed: metric_x_detailed.replace('_', ' ').title(),
                    metric_y_detailed: metric_y_detailed.replace('_', ' ').title()
                }
            )
            
            # Personalizar el gráfico agrupado
            fig_grouped.update_traces(
                textposition='top center',
                marker=dict(size=20)
            )
            
            # Añadir líneas de referencia para promedios
            fig_grouped.add_hline(
                y=df_filtered[metric_y_detailed].mean(),
                line_dash="dash",
                line_color="gray",
                opacity=0.5
            )
            fig_grouped.add_vline(
                x=df_filtered[metric_x_detailed].mean(),
                line_dash="dash",
                line_color="gray",
                opacity=0.5
            )
            
            # Actualizar layout
            fig_grouped.update_layout(
                showlegend=True,
                legend_title_text='Género',
                hovermode='closest'
            )
            
            st.plotly_chart(fig_grouped, use_container_width=True)

        # Añadir estadísticas descriptivas
        with st.expander("📊 Ver Estadísticas Descriptivas por Género"):
            stats_df = df_filtered.groupby('genre').agg({
                metric_x_detailed: ['mean', 'std', 'min', 'max'],
                metric_y_detailed: ['mean', 'std', 'min', 'max']
            }).round(2)
            
            # Formatear los nombres de las columnas
            stats_df.columns = [f"{col[0].replace('_', ' ').title()} - {col[1].title()}" 
                            for col in stats_df.columns]
            st.dataframe(stats_df, use_container_width=True)


    with tab5:
        st.header("📈 Evolución Temporal")

        # Selector de métricas para análisis temporal
        metrica_temporal = st.selectbox(
            "Selecciona la métrica a analizar en el tiempo:",
            [
                "Recaudación (Box Office)",
                "Presupuesto",
                "ROI",
                "Popularidad",
                "Puntuación Críticos",
                "Puntuación Audiencia"
            ],
            key="metrica_temporal"
        )

        # Mapeo de métricas a columnas
        mapeo_metricas_temporal = {
            "Recaudación (Box Office)": "box_office_millions",
            "Presupuesto": "budget_millions",
            "ROI": "roi",
            "Popularidad": "popularity",
            "Puntuación Críticos": "critic_score",
            "Puntuación Audiencia": "audience_score"
        }

        columna_temporal = mapeo_metricas_temporal[metrica_temporal]

        # Crear DataFrame con promedios anuales de forma más robusta
        df_temporal = df_filtered.groupby(['release_year', 'genre']).agg(
            mean=(columna_temporal, 'mean'),
            total=(columna_temporal, 'sum'),
            count=(columna_temporal, 'count')
        ).reset_index()

        # 4. Insights temporales
        st.subheader("🔍 Insights Temporales")
        
        # Calcular tendencias y puntos destacados
        mejor_anio = df_temporal.loc[df_temporal['mean'].idxmax()]
        peor_anio = df_temporal.loc[df_temporal['mean'].idxmin()]
        
        # Calcular tendencia general
        años = df_temporal['release_year'].unique()
        primer_año = años.min()
        ultimo_año = años.max()
        
        valor_inicial = df_temporal[df_temporal['release_year'] == primer_año]['mean'].mean()
        valor_final = df_temporal[df_temporal['release_year'] == ultimo_año]['mean'].mean()
        cambio_porcentual = ((valor_final - valor_inicial) / valor_inicial) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Mejor Año",
                f"{int(mejor_anio['release_year'])}",
                f"{mejor_anio['mean']:.1f} (promedio)"
            )
        
        with col2:
            st.metric(
                "Peor Año",
                f"{int(peor_anio['release_year'])}",
                f"{peor_anio['mean']:.1f} (promedio)"
            )
        
        with col3:
            st.metric(
                "Cambio Total",
                f"{cambio_porcentual:.1f}%",
                f"({primer_año} - {ultimo_año})"
            )
        
        # Análisis de tendencias por género
        st.markdown("### 📊 Tendencias por Género")
        
        # Calcular tendencias por género
        tendencias_genero = {}
        for genero in df_filtered['genre'].unique():
            datos_genero = df_temporal[df_temporal['genre'] == genero]
            if len(datos_genero) >= 2:  # Asegurar que hay suficientes datos
                x = datos_genero['release_year']
                y = datos_genero['mean']
                slope, intercept = np.polyfit(x, y, 1)
                tendencias_genero[genero] = slope
        
        # Ordenar géneros por tendencia
        generos_ordenados = sorted(
            tendencias_genero.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Mostrar tendencias
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Géneros en Ascenso:**")
            for genero, pendiente in generos_ordenados[:3]:
                st.markdown(f"- {genero}: {pendiente:+.2f} por año")
        
        with col2:
            st.markdown("**Géneros en Descenso:**")
            for genero, pendiente in generos_ordenados[-3:]:
                st.markdown(f"- {genero}: {pendiente:+.2f} por año")

        # 3. Análisis de tendencias
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Tendencia General")
            # Calcular tendencia general por año
            tendencia_general = df_filtered.groupby('release_year').agg(
                mean=(columna_temporal, 'mean'),
                median=(columna_temporal, 'median'),
                std=(columna_temporal, 'std')
            ).reset_index()
            
            fig_tendencia = px.line(
                tendencia_general,
                x='release_year',
                y=['mean', 'median'],
                title=f'Tendencia General de {metrica_temporal}',
                labels={
                    'release_year': 'Año',
                    'value': f'{metrica_temporal}',
                    'variable': 'Métrica'
                }
            )
            
            # Añadir área de desviación estándar
            fig_tendencia.add_traces(
                px.line(
                    tendencia_general,
                    x='release_year',
                    y=tendencia_general['mean'] + tendencia_general['std']
                ).data[0].update(line=dict(color='rgba(0,0,0,0)'))
            )
            
            fig_tendencia.add_traces(
                px.line(
                    tendencia_general,
                    x='release_year',
                    y=tendencia_general['mean'] - tendencia_general['std']
                ).data[0].update(line=dict(color='rgba(0,0,0,0)'))
            )
            
            st.plotly_chart(fig_tendencia, use_container_width=True)

        with col2:
            st.subheader("📊 Distribución por Década")
            # Añadir columna de década
            df_filtered['decade'] = (df_filtered['release_year'] // 10 * 10).astype(str) + 's'
            
            fig_decadas = px.box(
                df_filtered,
                x='decade',
                y=columna_temporal,
                title=f'Distribución de {metrica_temporal} por Década',
                labels={
                    'decade': 'Década',
                    columna_temporal: metrica_temporal
                }
            )
            
            st.plotly_chart(fig_decadas, use_container_width=True)

        # 2. Heatmap de evolución temporal
        st.subheader("🎨 Mapa de Calor Temporal")

        # Pivot table para el heatmap
        heatmap_data = df_temporal.pivot(
            index='genre',
            columns='release_year',
            values='mean'
        )

        fig_heatmap = px.imshow(
            heatmap_data,
            title=f'Mapa de Calor: {metrica_temporal} por Género y Año',
            labels=dict(x="Año", y="Género", color=metrica_temporal),
            aspect="auto",
            color_continuous_scale="Viridis"
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

        # 1. Gráfico de líneas por género a lo largo del tiempo
        st.subheader("📊 Evolución por Género")

        fig_lineas = px.line(
            df_temporal,
            x='release_year',
            y='mean',
            color='genre',
            title=f'Evolución de {metrica_temporal} por Género a lo largo del tiempo',
            labels={
                'release_year': 'Año',
                'mean': f'{metrica_temporal} (Promedio)',
                'genre': 'Género'
            }
        )

        fig_lineas.update_layout(
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig_lineas, use_container_width=True)


    with tab6:
            st.title("🎬 Sistema de Recomendación de Películas")
            st.write("### 🎯 Encuentra tu próxima película favorita")
            
            col1, col2 = st.columns(2)
            
            with col1:
                genero = st.selectbox(
                    "🎭 Género preferido:",
                    options=sorted(df_filtered['genre'].unique()),
                    key="genero_recomendador"  # Key única añadida
                )
                
                año_actual = int(df_filtered['release_year'].max())
                min_year = st.slider(
                    "📅 ¿Desde qué año?",
                    min_value=int(df_filtered['release_year'].min()),
                    max_value=año_actual,
                    value=año_actual - 10,
                    format="%d",
                    key="slider_año_recomendador"  # Key única añadida
                )
            
            with col2:
                popularidad_min = st.slider(
                    "⭐ Popularidad mínima:",
                    min_value=0,
                    max_value=10,
                    value=5,
                    format="%d",
                    key="slider_popularidad_recomendador"  # Key única añadida
                )
                
                n_recomendaciones = st.slider(
                    "🎯 Número de recomendaciones:",
                    min_value=1,
                    max_value=10,
                    value=5,
                    key="slider_num_recomendaciones"  # Key única añadida
                )

            if st.button("🔍 Buscar Recomendaciones", key="boton_recomendar"):
                with st.spinner("Buscando las mejores películas para ti..."):
                    recomendaciones, filtros_relajados = recomendar_peliculas(
                        df_filtered,
                        genero,
                        min_year,
                        popularidad_min,
                        n_recomendaciones
                    )
                    
                    if recomendaciones is None:
                        st.error("No se encontraron películas. Por favor, ajusta los criterios de búsqueda.")
                        
                        # Mostrar estadísticas útiles
                        st.markdown("### 📊 Estadísticas útiles para este género:")
                        genero_stats = df_filtered[df_filtered['genre'] == genero].agg({
                            'release_year': ['min', 'max'],
                            'popularity': ['min', 'max', 'mean']
                        })
                        
                        st.write(f"""
                        - Rango de años disponible: {int(genero_stats['release_year']['min'])} - {int(genero_stats['release_year']['max'])}
                        - Rango de popularidad: {genero_stats['popularity']['min']:.1f} - {genero_stats['popularity']['max']:.1f}
                        - Popularidad promedio: {genero_stats['popularity']['mean']:.1f}
                        """)
                    else:
                        if filtros_relajados:
                            st.info(f"💡 Se ajustaron los siguientes criterios para encontrar más coincidencias: {', '.join(filtros_relajados)}")
                        
                        st.success(f"¡Encontré {len(recomendaciones)} películas que podrían gustarte!")
                        
                        # Mostrar las recomendaciones
                        for idx, pelicula in recomendaciones.iterrows():
                            st.markdown(f"""
                            <div style="
                                padding: 20px;
                                border-radius: 10px;
                                background-color: #f0f2f6;
                                margin-bottom: 20px;
                                border-left: 5px solid #1f77b4;
                            ">
                                <h3 style="color: #1f77b4; margin-bottom: 10px;">
                                    {idx + 1}. {pelicula['title']} ({int(pelicula['release_year'])})
                                </h3>
                                <div style="display: grid; grid-template-columns: 1fr 1fr;">
                                    <div>
                                        🎭 <b>Género:</b> {pelicula['genre']}<br>
                                        ⭐ <b>Popularidad:</b> {pelicula['popularity']:.1f}/100<br>
                                        💰 <b>Taquilla:</b> ${pelicula['box_office_millions']:.1f}M
                                    </div>
                                    <div>
                                        👥 <b>Puntuación Audiencia:</b> {pelicula['audience_score']:.1f}/100<br>
                                        🎬 <b>Puntuación Críticos:</b> {pelicula['critic_score']:.1f}/100<br>
                                        📈 <b>ROI:</b> {pelicula['roi']:.1f}%
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Mostrar resumen estadístico
                        with st.expander("📊 Ver estadísticas de las recomendaciones"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Popularidad Promedio",
                                    f"{recomendaciones['popularity'].mean():.1f}/100"
                                )
                            with col2:
                                st.metric(
                                    "Puntuación Promedio",
                                    f"{((recomendaciones['critic_score'] + recomendaciones['audience_score'])/2).mean():.1f}/100"
                                )
                            with col3:
                                st.metric(
                                    "ROI Promedio",
                                    f"{recomendaciones['roi'].mean():.1f}%"
                                )
                        
                        # Análisis detallado expandible
                        with st.expander("📊 Ver Análisis Detallado de las Recomendaciones"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Gráfico de dispersión de puntuaciones
                                fig_scores = px.scatter(
                                    recomendaciones,
                                    x='critic_score',
                                    y='audience_score',
                                    size='popularity',
                                    color='roi',
                                    hover_data=['title'],
                                    title='Distribución de Puntuaciones'
                                )
                                st.plotly_chart(fig_scores)
                            
                            with col2:
                                # Gráfico de barras de popularidad
                                fig_pop = px.bar(
                                    recomendaciones,
                                    x='title',
                                    y='popularity',
                                    color='roi',
                                    title='Popularidad de las Películas Recomendadas'
                                )
                                fig_pop.update_xaxes(tickangle=45)
                                st.plotly_chart(fig_pop)
                            
                            # Tabla de datos completa
                            st.write("### 📋 Datos Detallados")
                            st.dataframe(
                                recomendaciones[[
                                    'title', 'release_year', 'genre', 
                                    'critic_score', 'audience_score', 'popularity',
                                    'box_office_millions', 'budget_millions', 'roi'
                                ]].style.background_gradient(subset=['popularity', 'roi'], cmap='YlOrRd')
                            )
