"""
Este módulo sirve como punto de entrada principal para la aplicación de análisis de películas.
Actúa como una capa de abstracción entre el código de la aplicación principal y el dashboard,
facilitando la gestión y el mantenimiento del código.

Características principales:
-------------------------
- Inicializa y ejecuta la aplicación principal
- Gestiona la importación y ejecución del dashboard
- Proporciona un punto de entrada único y limpio
- Facilita la separación de responsabilidades entre módulos

Dependencias:
-----------
- dashboard.py: Contiene la implementación del dashboard y la lógica de visualización
- analysis.py: Contiene las funciones de análisis de datos
- visualization.py: Contiene las funciones de generación de gráficos

Uso:
----
Para ejecutar la aplicación:
    $ python app.py

La aplicación iniciará un servidor local de Streamlit con el dashboard interactivo.

Notas:
-----
- Asegúrese de tener todas las dependencias instaladas antes de ejecutar
- El archivo requirements.txt contiene todas las dependencias necesarias
- La aplicación espera encontrar los módulos dashboard.py, analysis.py y visualization.py
  en el mismo directorio

"""

from dashboard import mi_codigo

def main():
    """
    Función principal que inicializa y ejecuta la aplicación.
    Sirve como punto de entrada único para todo el sistema.
    
    Esta función:
    1. Importa el código necesario del módulo dashboard
    2. Ejecuta la función principal del dashboard
    3. Maneja cualquier configuración inicial necesaria
    
    Returns:
        None
    """
    mi_codigo()

if __name__ == "__main__":
    main()
