pip install -r requirements.txt

# 🌡️ Predicción de Temperatura en Santiago
### Proyecto Semestral – Ciencia de Datos / Python + Streamlit + ML

Este proyecto desarrolla una aplicación interactiva para predecir la temperatura mínima y máxima en Santiago de Chile usando modelos de *Machine Learning* y *Deep Learning*.


La app permite analizar datos históricos, entrenar modelos y realizar predicciones en tiempo real mediante:

- 🔥 **LSTM (Red Neuronal Recurrente)**

---

## 🚀 Tecnologías utilizadas

- **Python 3.12**
- **Streamlit**
- **PyTorch**
- **Scikit-learn**
- **Pandas / NumPy**
- **Joblib**
- **Plotly**

---

## 🧠 Modelos entrenados

Los modelos se entrenan con datos meteorológicos reales de Santiago (enero-noviembre 2025), usando principalmente:

| Variable         | Descripción                        |
|------------------|------------------------------------|
| tMin24Horas      | Temp. mínima últimas 24 horas (°C) |
| tMax24Horas      | Temp. máxima últimas 24 horas (°C) |
| momento          | Fecha y hora                       |


Modelo generado:

- `lstm_temp_predictor.pth` (LSTM)

---

## 📂 Estructura del proyecto

📁 proyecto_semestral
│── app.py
│── requirements.txt
│── README.md
│── data/
│   └── *.csv, temperatura_limpia_normalizada.pkl, lstm_temp_predictor.pth
│── ui/
│   └── ui_home.py, ui_analitica_web.py, ui_modelo.py, ui_entrenamiento.py, ui_preprocesamiento.py
│── docs/
│   └── narrativa.md, instrucciones.md, prompts_ia.md

---
## ▶️ Flujo del proyecto

1. Preprocesar los datos (opcional, si tienes los .csv originales)
2. Entrenar el modelo LSTM desde la app (sección "Preprocesar y Entrenar")
3. Analizar los datos y visualizar gráficos (sección "Análisis de datos")
4. Realizar predicciones de temperatura (sección "Predicción")

## 💡 Acerca del proyecto

El objetivo es crear un sistema predictivo accesible y moderno para estimar la temperatura en Santiago, demostrando:

- Procesamiento y limpieza de datos
- Entrenamiento de modelos ML
- Integración con Streamlit
- Visualización interactiva con Plotly
- Interpretación de resultados

---
## 📊 Modelos utilizados


🔹 LSTM (PyTorch)
	- Predicción de temperatura mínima y máxima usando series temporales.

## 🧪 Evaluación del desempeño

Se usan las siguientes métricas:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- R² (Coeficiente de determinación)

Los resultados se muestran en la app tras el entrenamiento.

---
## 👥 Integrantes del equipo

**Helen Maureira - Francisco Provoste**
Estudiantes de Ciencia de Datos
Universidad Tecnológica Metropolitana (UTEM)
2025

---

## 📜 Licencia

Proyecto de uso académico.
