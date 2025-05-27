# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import tempfile

# ---------------------------
# Función para cargar desde GitHub main
# ---------------------------
@st.cache_resource
def cargar_modelo_desde_url(url):
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(response.content)
        return joblib.load(tmp_file.name)

# URLs de los modelos desde el branch main
base_url = "https://raw.githubusercontent.com/Emwerick/ProjFinML_EE/main/"
url_mdlRegLin = base_url + "mdlRegLin.pkl"
url_mdlSVM = base_url + "mdlSVM.pkl"
url_mdlRanFor = base_url + "mdlRanFor.pkl"
url_mdlBagging = base_url + "mdlBagging.pkl"
url_preproc = base_url + "preprocessor.pkl"

# Cargar modelos
mdlRegLin = cargar_modelo_desde_url(url_mdlRegLin)
mdlSVM = cargar_modelo_desde_url(url_mdlSVM)
mdlRanFor = cargar_modelo_desde_url(url_mdlRanFor)
mdlBagging = cargar_modelo_desde_url(url_mdlBagging)
preprocessor = cargar_modelo_desde_url(url_preproc)

# ---------------------------
# Interfaz Streamlit
# ---------------------------
st.title('Predicción de SPEI12')

# Selección del modelo
modelo_seleccionado = st.selectbox(
    'Seleccionar el modelo a usar en la predicción:',
    ['Regresión lineal', 'SVM', 'Random Forest', 'Ensamble Bagging con regresión lineal']
)

# Selector de mes y año
nombres_meses = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
año = st.selectbox("Año", list(range(1954, 2035)), index=71)
mes_nombre = st.selectbox("Mes", nombres_meses)

# Inputs SPEI 1-11
spei_inputs = []
for i in range(1, 12):
    val = st.number_input(f'SPEI_{i}', format="%.3f")
    spei_inputs.append(val)

# Construir el dataframe de entrada
data_str = f"{mes_nombre}{año}"
input_dict = {'DATA': [data_str]}
for i, val in zip(range(1, 12), spei_inputs):
    input_dict[f'SPEI_{i}'] = [val]
for i in range(1, 35):
    input_dict[f'V{i}'] = [np.nan]

df_input = pd.DataFrame(input_dict)

st.subheader("DataFrame de entrada estructurado")
st.write(df_input)

# Preprocesamiento
try:
    X_transformado = preprocessor.transform(df_input)
    st.subheader("Datos transformados")
    st.write(X_transformado)
except Exception as e:
    st.error(f"Error en la transformación: {e}")
    st.stop()

# Botón de predicción
if st.button('Predecir'):
    try:
        if modelo_seleccionado == 'Regresión lineal':
            prediccion = mdlRegLin.predict(X_transformado)
        elif modelo_seleccionado == 'SVM':
            prediccion = mdlSVM.predict(X_transformado)
        elif modelo_seleccionado == 'Random Forest':
            prediccion = mdlRanFor.predict(X_transformado)
        else:
            prediccion = mdlBagging.predict(X_transformado)

        st.success(f'La predicción es: {prediccion[0]:.3f}')

    except Exception as e:
        st.error(f'Error en la predicción: {e}')
