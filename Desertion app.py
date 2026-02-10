import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Predicción de Deserción Estudiantil",
    layout="wide"
)

st.title("Sistema de Predicción de Deserción Estudiantil")
st.write("Aplicación basada en Machine Learning para identificar estudiantes en riesgo académico")

model = joblib.load("modelo_desercion.pkl")
scaler = joblib.load("scaler.pkl")

df = pd.read_excel("REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.xlsx")
df["DESERCION"] = df["ESTADO"].apply(lambda x: 1 if x == "REPROBADA" else 0)

menu = st.sidebar.radio(
    "Menú",
    ["Inicio", "Análisis Exploratorio", "Métricas del Modelo", "Predicción"]
)

if menu == "Inicio":
    st.subheader("Descripción del Proyecto")

    st.write("""
    Este sistema utiliza técnicas de minería de datos y Machine Learning para 
    predecir el riesgo de deserción estudiantil a partir de información académica.

    Variables utilizadas:
    - Asistencia
    - Número de veces que cursa la materia
    - Nivel académico
    """)

    st.success("Modelo entrenado con Random Forest — Precisión: 88%")

elif menu == "Análisis Exploratorio":
    st.subheader("Análisis Exploratorio de Datos")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Distribución de Asistencia")
        fig, ax = plt.subplots()
        sns.histplot(df["ASISTENCIA"], bins=20, ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("### Distribución de Nivel")
        fig, ax = plt.subplots()
        sns.histplot(df["NIVEL"], bins=10, ax=ax)
        st.pyplot(fig)

    st.write("### Relación entre asistencia y deserción")
    fig, ax = plt.subplots()
    sns.boxplot(x="DESERCION", y="ASISTENCIA", data=df, ax=ax)
    st.pyplot(fig)

elif menu == "Métricas del Modelo":
    st.subheader("Resultados del Modelo")

    st.metric("Accuracy", "88%")
    st.metric("Recall (Deserción)", "78%")
    st.metric("Precisión (Deserción)", "63%")

    st.write("### Matriz de Confusión")
    matriz = np.array([[663, 71], [35, 121]])
    fig, ax = plt.subplots()
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    st.pyplot(fig)

    st.write("### Importancia de Variables")
    variables = ["ASISTENCIA", "NIVEL", "NO. VEZ"]
    importancias = [0.857, 0.101, 0.042]

    fig, ax = plt.subplots()
    sns.barplot(x=importancias, y=variables, ax=ax)
    st.pyplot(fig)

elif menu == "Predicción":
    st.subheader("Predicción de Riesgo")

    asistencia = st.slider("Asistencia (%)", 0, 100, 80)
    num_vez = st.number_input("Número de veces que cursa la materia", 1, 5, 1)
    nivel = st.number_input("Nivel académico", 1, 10, 1)

    if st.button("Predecir riesgo"):
        datos = np.array([[asistencia, num_vez, nivel]])
        datos_scaled = scaler.transform(datos)
        pred = model.predict(datos_scaled)

        if pred[0] == 1:
            st.error("Riesgo ALTO de deserción")
        else:
            st.success("Riesgo BAJO de deserción")
