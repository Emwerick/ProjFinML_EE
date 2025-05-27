# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import tempfile
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Predicción SPEI12", layout="wide")

# ---------------------------
# URLs de modelos desde Releases (GitHub)
# ---------------------------
url_mdlRegLin = "https://github.com/Emwerick/ProjFinML_EE/releases/download/v1.0/mdlRegLin.pkl"
url_mdlSVM = "https://github.com/Emwerick/ProjFinML_EE/releases/download/v1.0/mdlSVM.pkl"
url_mdlRanFor = "https://github.com/Emwerick/ProjFinML_EE/releases/download/v1.0/mdlRanFor.pkl"
url_mdlBagging = "https://github.com/Emwerick/ProjFinML_EE/releases/download/v1.0/mdlBagging.pkl"

# Imagen portada desde la rama principal
URL_PORTADA = "https://raw.githubusercontent.com/Emwerick/ProjFinML_EE/main/Portada.png"

# ---------------------------
# Cargar modelos desde releases
# ---------------------------
@st.cache_resource
def cargar_modelo_desde_url(url):
    response = requests.get(url)
    if response.status_code != 200 or len(response.content) < 1000:
        raise ValueError(f"Error al descargar modelo desde {url} - tamaño inesperado.")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(response.content)
        return joblib.load(tmp_file.name)

# Cargar modelos
mdlRegLin = cargar_modelo_desde_url(url_mdlRegLin)
mdlSVM = cargar_modelo_desde_url(url_mdlSVM)
mdlRanFor = cargar_modelo_desde_url(url_mdlRanFor)
mdlBagging = cargar_modelo_desde_url(url_mdlBagging)

# ---------------------------
# Interfaz
# ---------------------------
st.title("Predicción de SPEI12")
st.image(URL_PORTADA, use_container_width=True)

with st.expander("Descripción del proyecto"):
    st.markdown("""
    El índice SPEI mide el nivel de sequía de una región específica. Esta aplicación predice el índice **SPEI12**
    para un mes y año específico, a partir de las observaciones SPEI1 al SPEI11, utilizando modelos entrenados.
    """)

# Sidebar
st.sidebar.header("📋 Parámetros de entrada")

modelo_seleccionado = st.sidebar.selectbox(
    'Modelo de predicción:',
    ['Regresión lineal', 'SVM', 'Random Forest', 'Ensamble Bagging con regresión lineal']
)

nombres_meses = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
año = st.sidebar.selectbox("Año", list(range(1954, 2035)), index=71)
mes_nombre = st.sidebar.selectbox("Mes", nombres_meses)

spei_inputs = []
for i in range(1, 12):
    val = st.sidebar.number_input(f'SPEI_{i}', format="%.3f")
    spei_inputs.append(val)

# ---------------------------
# Construcción del DataFrame
# ---------------------------
data_str = f"{mes_nombre}{año}"
input_dict = {'DATA': [data_str]}
for i, val in zip(range(1, 12), spei_inputs):
    input_dict[f'SPEI_{i}'] = [val]
for i in range(1, 35):
    input_dict[f'V{i}'] = [np.nan]
df_input = pd.DataFrame(input_dict)

# ---------------------------
# Preprocesamiento
# ---------------------------
def preprocesar_input(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['DATA'], format='%b%Y')
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    num_cols = [f"SPEI_{i}" for i in range(1, 12)]
    num_data = df[num_cols].values
    num_data = SimpleImputer(strategy="mean").fit_transform(num_data)
    num_data = StandardScaler().fit_transform(num_data)

    date_features = df[['month', 'month_sin', 'month_cos', 'year']].values
    date_features = StandardScaler().fit_transform(date_features)

    return np.hstack([num_data, date_features])

# ---------------------------
# Botón de predicción
# ---------------------------
st.markdown("---")
if st.button("🔍 Predecir SPEI12"):
    try:
        X_transformado = preprocesar_input(df_input)

        if modelo_seleccionado == 'Regresión lineal':
            prediccion = mdlRegLin.predict(X_transformado)
        elif modelo_seleccionado == 'SVM':
            prediccion = mdlSVM.predict(X_transformado)
        elif modelo_seleccionado == 'Random Forest':
            prediccion = mdlRanFor.predict(X_transformado)
        else:
            prediccion = mdlBagging.predict(X_transformado)

        valor_predicho = float(prediccion[0])

        etiquetas = [f'SPEI_{i}' for i in range(1, 12)] + ['SPEI_12 (Predicho)']
        valores = spei_inputs + [valor_predicho]
        colores = ['mediumseagreen'] * 11 + ['orange']

        fig = go.Figure(data=[
            go.Bar(x=etiquetas, y=valores, marker_color=colores)
        ])
        fig.update_layout(
            title="Valores de SPEI (1-11) y Predicción de SPEI_12",
            xaxis_title="Mes",
            yaxis_title="Valor",
            title_x=0.5,
            shapes=[dict(type="line", x0=-0.5, x1=len(etiquetas)-0.5, y0=0, y1=0,
                         line=dict(color="gray", dash="dash"))]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"✅ Predicción del SPEI12: **{valor_predicho:.3f}**")

    except Exception as e:
        st.error(f"Error en la predicción: {e}")
