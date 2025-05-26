import streamlit as st
import joblib
import requests
import numpy as np
import pandas as pd
from io import BytesIO

# Título de la app
st.set_page_config(page_title="Predicción de SPEI12", layout="centered")
st.title('🌵 Predicción de SPEI12')

with st.expander("Descripción del proyecto"):
    st.markdown("""
    Esta aplicación predice el valor del **SPEI12** (Índice de Precipitación Estándar a 12 meses) usando diferentes modelos de regresión:
    - Regresión Lineal
    - Soporte Vectorial (SVM)
    - Random Forest
    - Ensamble Bagging con Regresión Lineal

    Los modelos fueron entrenados previamente y están disponibles en un repositorio de GitHub.
    """)

# Función para cargar modelos desde GitHub
@st.cache_resource
def cargar_modelo_desde_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return joblib.load(BytesIO(response.content))
    else:
        st.error(f"No se pudo cargar desde GitHub: {url}")
        return None

# URLs base de los modelos
base_url = "https://raw.githubusercontent.com/Emwerick/ProjFinML_EE/main/"
preprocessor = cargar_modelo_desde_github(base_url + "preprocessor.pkl")
mdlRegLin = cargar_modelo_desde_github(base_url + "mdlRegLin.pkl")
mdlSVM = cargar_modelo_desde_github(base_url + "mdlSVM.pkl")
mdlRanFor = cargar_modelo_desde_github(base_url + "mdlRanFor.pkl")
mdlBagging = cargar_modelo_desde_github(base_url + "mdlBagging.pkl")

# Selector de modelo
modelo_seleccionado = st.selectbox('🔍 Seleccionar el modelo a usar:', [
    'Regresión lineal', 'SVM', 'Random Forest', 'Bagging'])

# Selector de fecha
nombres_meses = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
año = st.selectbox("Año", list(range(1954, 2035)), index=71)
mes_nombre = st.selectbox("Mes", nombres_meses)
data_str = f"{mes_nombre}{año}"

# Inputs SPEI
spei_inputs = []
for i in range(1, 12):
    val = st.number_input(f'SPEI_{i}', format="%.3f")
    spei_inputs.append(val)

# Crear DataFrame
input_dict = {'DATA': [data_str]}
for i, val in zip(range(1, 12), spei_inputs):
    input_dict[f'SPEI_{i}'] = [val]
for i in range(1, 35):
    input_dict[f'V{i}'] = [np.nan]
df_input = pd.DataFrame(input_dict)

st.subheader("📄 Vista previa de entrada")
st.dataframe(df_input)

# Transformar datos
try:
    X_transformado = preprocessor.transform(df_input)
    st.subheader("🔄 Datos transformados")
    st.dataframe(pd.DataFrame(X_transformado))
except Exception as e:
    st.error(f"Error al transformar datos: {e}")

# Botón de predicción
if st.button('🚀 Predecir'):
    try:
        if modelo_seleccionado == 'Regresión lineal':
            prediccion = mdlRegLin.predict(X_transformado)
        elif modelo_seleccionado == 'SVM':
            prediccion = mdlSVM.predict(X_transformado)
        elif modelo_seleccionado == 'Random Forest':
            prediccion = mdlRanFor.predict(X_transformado)
        else:
            prediccion = mdlBagging.predict(X_transformado)

        st.success(f'✅ La predicción del SPEI12 es: {prediccion[0]:.3f}')

    except Exception as e:
        st.error(f'❌ Error en la predicción: {e}')
