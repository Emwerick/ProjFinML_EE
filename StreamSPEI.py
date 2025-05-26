# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Predicción de SPEI12", layout="centered")

st.title("🌧️ Predicción de SPEI12")

# Descripción del proyecto
with st.expander("📝 Descripción del proyecto"):
    st.markdown("""
    Esta aplicación permite predecir el valor del índice **SPEI12** (Índice de Precipitación Estandarizado a 12 meses) a partir de los valores de SPEI pasados (SPEI_1 a SPEI_11), mes y año.

    Se utilizan distintos modelos de aprendizaje automático previamente entrenados:

    - Regresión Lineal
    - SVM
    - Random Forest
    - Ensamble Bagging

    El modelo seleccionado realizará la predicción una vez ingresados los valores.
    """)

# Cargar modelos desde GitHub
@st.cache_resource
def cargar_modelo_desde_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return joblib.load(BytesIO(response.content))
    else:
        st.error(f"No se pudo cargar el modelo desde GitHub: {url}")
        return None

base_url = "https://raw.githubusercontent.com/Emwerick/ProjFinML_EE/main/"

mdlRegLin = cargar_modelo_desde_github(base_url + "mdlRegLin.pkl")
mdlSVM = cargar_modelo_desde_github(base_url + "mdlSVM.pkl")
mdlRanFor = cargar_modelo_desde_github(base_url + "mdlRanFor.pkl")
mdlBagging = cargar_modelo_desde_github(base_url + "mdlBagging.pkl")
preprocessor = cargar_modelo_desde_github(base_url + "preprocessor.pkl")

# Selección del modelo
modelo_seleccionado = st.selectbox("Seleccionar el modelo a usar en la predicción:", [
    "Regresión lineal", "SVM", "Random Forest", "Ensamble Bagging con regresión lineal"])

# Inputs de mes y año
nombres_meses = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
año = st.selectbox("Año", list(range(1954, 2035)), index=71)
mes_nombre = st.selectbox("Mes", nombres_meses)

# Inputs SPEI
spei_inputs = []
for i in range(1, 12):
    val = st.number_input(f"SPEI_{i}", format="%.3f")
    spei_inputs.append(val)

# Construcción del DataFrame de entrada
data_str = f"{mes_nombre}{año}"
input_dict = {'DATA': [data_str]}
for i, val in zip(range(1, 12), spei_inputs):
    input_dict[f"SPEI_{i}"] = [val]
for i in range(1, 35):
    input_dict[f"V{i}"] = [np.nan]

df_input = pd.DataFrame(input_dict)

st.subheader("📋 Datos de entrada")
st.dataframe(df_input)

# Transformar datos
try:
    X_transformado = preprocessor.transform(df_input)
    st.subheader("🔄 Datos transformados")
    st.dataframe(pd.DataFrame(X_transformado))
except Exception as e:
    st.error(f"Error en el preprocesamiento: {e}")

# Predicción
if st.button("Predecir"):
    try:
        if modelo_seleccionado == 'Regresión lineal':
            prediccion = mdlRegLin.predict(X_transformado)
        elif modelo_seleccionado == 'SVM':
            prediccion = mdlSVM.predict(X_transformado)
        elif modelo_seleccionado == 'Random Forest':
            prediccion = mdlRanFor.predict(X_transformado)
        else:
            prediccion = mdlBagging.predict(X_transformado)

        st.success(f"🌟 La predicción de SPEI12 es: {prediccion[0]:.3f}")
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
