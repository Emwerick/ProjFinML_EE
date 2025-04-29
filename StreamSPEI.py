# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Cargar los modelos
mdlRegLin = joblib.load('C:/Users/erick/Documentos/Maestria UACH/Machine Learning/F1/mdlRegLin.pkl')
mdlSVM = joblib.load('C:/Users/erick/Documentos/Maestria UACH/Machine Learning/F2/mdlSVM.pkl')
mdlRanFor = joblib.load('C:/Users/erick/Documentos/Maestria UACH/Machine Learning/F3/mdlRanFor.pkl')

# Título de la app
st.title('Predicción de SPEI12')

# Elegir el modelo
modelo_seleccionado = st.selectbox('Seleccionar el modelo a usar en la predicción:', ['Regresión lineal', 'SVM', 'Random Forest'])

# Inputs para la predicción

def fecha_sen_cos(X):
    X = X.copy()
    X['date'] = pd.to_datetime(X['DATA'], format='%b%Y')
    X['month'] = X['date'].dt.month 
    X['year'] = X['date'].dt.year

    X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
    X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)

    return X[['month', 'month_sin', 'month_cos', 'year']]

preprocessor = joblib.load('C:/Users/erick/Documentos/Maestria UACH/Machine Learning/F3/preprocessor.pkl')  # Usa el path relativo o absoluto correcto

# Selector de año y mes
nombres_meses = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]
año = st.selectbox("Año", list(range(1954, 2035)), index=71)
mes_nombre = st.selectbox("Mes", nombres_meses)
#mes_numero = nombres_meses.index(mes_nombre) + 1

# Inputs SPEI
spei_inputs = []
for i in range(1, 12):
    val = st.number_input(f'SPEI_{i}', format="%.3f")
    spei_inputs.append(val)

data_str = f"{mes_nombre}{año}"

input_dict = {'DATA': [data_str]}
for i, val in zip(range(1, 12), spei_inputs):
    input_dict[f'SPEI_{i}'] = [val]
    
for i in range(1, 35):
    input_dict[f'V{i}'] = [np.nan]
    
df_input = pd.DataFrame(input_dict)

st.subheader("DataFrame de entrada estructurado")
st.write(df_input)

X_transformado = preprocessor.transform(df_input)

st.subheader("Datos transformados")
st.write(X_transformado)

# Botón para predecir
if st.button('Predecir'):
    # Realiza la transformación (esto debe estar ya hecho previamente)
    try:
        X_transformado = preprocessor.transform(df_input)
        
        # Selección del modelo
        if modelo_seleccionado == 'Regresión lineal':
            prediccion = mdlRegLin.predict(X_transformado)
        elif modelo_seleccionado == 'SVM':
            prediccion = mdlSVM.predict(X_transformado)
        else:
            prediccion = mdlRanFor.predict(X_transformado)
        
        st.success(f'La predicción es: {prediccion[0]:.3f}')
    
    except Exception as e:
        st.error(f'Error en la predicción: {e}')
