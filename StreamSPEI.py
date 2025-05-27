# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dill
import requests
import io

st.set_page_config(page_title="Predicción SPEI12", layout="wide")

# Función para cargar archivos desde GitHub raw
def cargar_desde_github(url):
    response = requests.get(url)
    response.raise_for_status()
    return io.BytesIO(response.content)

# === URLs de GitHub raw ===
URL_REG_LIN = "https://raw.githubusercontent.com/Emwerick/ProjFinML_EE/main/mdlRegLin.pkl"
URL_SVM = "https://raw.githubusercontent.com/Emwerick/ProjFinML_EE/main/mdlSVM.pkl"
URL_RAN_FOR = "https://raw.githubusercontent.com/Emwerick/ProjFinML_EE/main/mdlRanFor.pkl"
URL_BAGGING = "https://raw.githubusercontent.com/Emwerick/ProjFinML_EE/main/mdlBagging.pkl"
URL_PREPROCESSOR = "https://raw.githubusercontent.com/Emwerick/ProjFinML_EE/main/preprocessor2.pkl"
URL_PORTADA = "https://raw.githubusercontent.com/Emwerick/ProjFinML_EE/main/Portada.png"

# Cargar modelos desde GitHub
mdlRegLin = joblib.load(cargar_desde_github(URL_REG_LIN))
mdlSVM = joblib.load(cargar_desde_github(URL_SVM))
mdlRanFor = joblib.load(cargar_desde_github(URL_RAN_FOR))
mdlBagging = joblib.load(cargar_desde_github(URL_BAGGING))

# Cargar preprocesador
with cargar_desde_github(URL_PREPROCESSOR) as f:
    preprocessor = dill.load(f)

# Título y portada
st.title("Predicción de SPEI12")
st.image(URL_PORTADA, use_container_width=True)

with st.expander("Descripción del proyecto"):
    st.markdown("""
    El índice SPEI mide el nivel de sequía de una región específica. Se basa en la información disponible de las precipitaciones 
    y temperatura. Combina características multiescalares junto con la habilidad de incorporar variabilidad de temperatura.

    Esta aplicación predice el índice **SPEI12** para un mes y año específico, a partir de las observaciones del índice SPEI1 al SPEI11. 
    Para realizar la predicción utiliza modelos previamente entrenados, incluyendo:

    - Regresión Lineal
    - SVM
    - Random Forest
    - Bagging con Regresión Lineal

    En la columna de la izquierda se debe ingresar la fecha deseada para la predicción, así como 
    los valores observados de SPEI1 - SPEI11.
    """)

# Barra lateral
st.sidebar.header("Parámetros de entrada")

modelo_seleccionado = st.sidebar.selectbox(
    'Modelo de predicción:',
    ['Regresión lineal', 'SVM', 'Random Forest', 'Ensamble Bagging con regresión lineal']
)

nombres_meses = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
año = st.sidebar.selectbox("Año", list(range(1954, 2035)), index=71)
mes_nombre = st.sidebar.selectbox("Mes", nombres_meses)

spei_inputs = [st.sidebar.number_input(f'SPEI_{i}', format="%.3f") for i in range(1, 12)]

data_str = f"{mes_nombre}{año}"
input_dict = {'DATA': [data_str]}
input_dict.update({f'SPEI_{i}': [val] for i, val in zip(range(1, 12), spei_inputs)})
input_dict.update({f'V{i}': [np.nan] for i in range(1, 35)})
df_input = pd.DataFrame(input_dict)

# Preprocesamiento
try:
    X_transformado = preprocessor.transform(df_input)
except Exception as e:
    st.error(f"Error en la transformación: {e}")
    st.stop()

# Botón de predicción
st.markdown("---")
if st.button("Predecir SPEI12"):
    try:
        if modelo_seleccionado == 'Regresión lineal':
            prediccion = mdlRegLin.predict(X_transformado)
        elif modelo_seleccionado == 'SVM':
            prediccion = mdlSVM.predict(X_transformado)
        elif modelo_seleccionado == 'Random Forest':
            prediccion = mdlRanFor.predict(X_transformado)
        else:
            prediccion = mdlBagging.predict(X_transformado)

        valor_predicho = float(prediccion[0])

        # Gráfico
        st.subheader("Visualización de datos de entrada")
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
            shapes=[
                dict(type="line", x0=-0.5, x1=len(etiquetas)-0.5, y0=0, y1=0,
                     line=dict(color="gray", dash="dash"))
            ]
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"Predicción del SPEI12: **{valor_predicho:.3f}**")

    except Exception as e:
        st.error(f"Error en la predicción: {e}")
