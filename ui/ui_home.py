# Página de presentación del proyecto en Streamlit
import streamlit as st

def main():
	st.set_page_config(page_title="Predicción de Temperatura en Santiago", page_icon="🌡️", layout="centered")
	st.title("Predicción de Temperatura en Santiago de Chile")
	st.markdown("""
Bienvenido al proyecto de **predicción de temperatura** para la ciudad de Santiago de Chile.

Este trabajo utiliza datos meteorológicos históricos y modelos de machine learning (incluyendo LSTM) para estimar la temperatura futura.

**Características principales:**
- Integración de datos reales desde enero a noviembre.
- Preprocesamiento y normalización automática de los datos.
- Entrenamiento interactivo de modelos en la web.
- Predicción personalizada por día y mes.
- Visualización de métricas y evolución del entrenamiento.

**¿Cómo usar la aplicación?**
1. Explora los datos y gráficos en la sección de análisis.
2. Entrena el modelo LSTM en la sección de entrenamiento.
3. Realiza predicciones ingresando los valores de las últimas 24 horas en la sección de predicción.

---
**Autores:** Helen Maureira, Francisco Provoste
**Fecha:** 10/12/2025
""")
