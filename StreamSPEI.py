# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# URL base de GitHub donde están los modelos
base_url = "https://raw.githubusercontent.com/Emwerick/ProjFinML_EE/main/"  # <-- REEMPLAZA esto con tu URL real

@st.cache_resource
def cargar_modelo_desde_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return joblib.load(BytesIO(response.content))
    else:
        raise Exception(f"No se pudo cargar el archivo desde: {url}")

# Cargar modelos desde GitHub
mdlRegLin = cargar_modelo_desde_github(base_url + "mdlRegLin.pkl")
mdlSVM = cargar_modelo_desde_github(base_url + "mdlSVM.pkl")
mdlRanFor = cargar_modelo_desde_github(base_url + "mdlRanFor.pkl")
mdlBagging = cargar_modelo_desde_github(base_url + "mdlBagging.pkl")

# Función de preprocesamiento
def preprocesar_input(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['DATA'], format='%b%Y')
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Columnas SPEI
    num_cols = [f"SPEI_{i}" for i in range(1, 12)]
    num_data = df[num_cols].values
    num_data = SimpleImputer(strategy="mean").fit_transform(num_data)
    num_data = StandardScaler().fit_transform(num_data)

    # Columnas de fecha transformadas
    date_features = df[['month', 'month_sin', 'month_cos', 'year']].values
    date_features = StandardScaler().fit_transform(date_features)

    return np.hstack([num_data, date_features])

# Título de la app
st.title('Predicción de SPEI12')

# Selección del modelo
modelo_seleccionado = st.selectbox(
    'Seleccionar el modelo a usar en la predicción:',
    ['Regresión lineal', 'SVM', 'Random Forest', 'Ensamble Bagging con regresión lineal']
)

# Selector de año y mes
nombres_meses = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
año = st.selectbox("Año", list(range(1954, 2035)), index=71)
mes_nombre = st.selectbox("Mes", nombres_meses)

# Inputs SPEI
spei_inputs = []
for i in range(1, 12):
    val = st.number_input(f'SPEI_{i}', format="%.3f")
    spei_inputs.append(val)

# Armar DataFrame de entrada
data_str = f"{mes_nombre}{año}"
input_dict = {'DATA': [data_str]}
for i, val in zip(range(1, 12), spei_inputs):
    input_dict[f'SPEI_{i}'] = [val]
for i in range(1, 35):
    input_dict[f'V{i}'] = [np.nan]

df_input = pd.DataFrame(input_dict)

# Mostrar DataFrame original
st.subheader("DataFrame de entrada estructurado")
st.write(df_input)

# Preprocesar entrada
try:
    X_transformado = preprocesar_input(df_input)
    st.subheader("Datos transformados")
    st.write(X_transformado)
except Exception as e:
    st.error(f"Error en la transformación: {e}")
    st.stop()

# Botón para predecir
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
