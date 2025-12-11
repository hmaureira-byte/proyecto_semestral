# Cambios principales:
# - ARIMA como modelo principal
# - Sección de modelos entrenados y flujo actualizado

# 🌡️ Predicción de Temperatura en Santiago
### Proyecto Semestral – Ciencia de Datos / Python + Streamlit + ML

Este proyecto desarrolla una aplicación interactiva para predecir la temperatura mínima y máxima en Santiago de Chile usando modelos de *Machine Learning* y *Series Temporales*.

La app permite analizar datos históricos, ajustar modelos y realizar predicciones en tiempo real mediante:

- 🔥 **ARIMA (Modelo de series temporales)**

---

## 🚀 Tecnologías utilizadas

- **Python 3.12**
- **Streamlit**
- **statsmodels**
- **Scikit-learn**
- **Pandas / NumPy**
- **Joblib**
- **Plotly**

---

## 🧠 Modelos entrenados

Los modelos se ajustan con datos meteorológicos reales de Santiago (enero-noviembre 2025), usando principalmente:

| Variable         | Descripción                        |
|------------------|------------------------------------|
| tMin24Horas      | Temp. mínima últimas 24 horas (°C) |
| tMax24Horas      | Temp. máxima últimas 24 horas (°C) |
| momento          | Fecha y hora                       |

Modelos generados:
- `arima_min_model.pkl` (ARIMA temperatura mínima)
- `arima_max_model.pkl` (ARIMA temperatura máxima)

---

## 📂 Estructura del proyecto

📁 proyecto_semestral
│── app.py
│── requirements.txt
│── README.md
│── data/
│   └── *.csv, temperatura_limpia.pkl, arima_min_model.pkl, arima_max_model.pkl
│── ui/
│   └── ui_home.py, ui_analitica_web.py, ui_modelo.py, ui_ajuste.py, ui_preprocesamiento.py
│── docs/
│   └── narrativa.md, instrucciones.md, prompts_ia.md

---
## ▶️ Flujo del proyecto

1. Preprocesar los datos (opcional, si tienes los .csv originales)
2. Ajustar el modelo ARIMA desde la app (sección "Ajuste de modelo")
3. Analizar los datos y visualizar gráficos (sección "Análisis de datos")
4. Realizar predicciones de temperatura (sección "Predicción")

## 💡 Acerca del proyecto

El objetivo es crear un sistema predictivo accesible y moderno para estimar la temperatura en Santiago, demostrando:

- Procesamiento y limpieza de datos
- Ajuste de modelos de series temporales
- Integración con Streamlit
- Visualización interactiva con Plotly
- Interpretación de resultados

---
## 📊 Modelos utilizados

🔹 ARIMA (statsmodels)
    - Predicción de temperatura mínima y máxima usando series temporales.

## 🧪 Evaluación del desempeño

Se usan las siguientes métricas:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- R² (Coeficiente de determinación)

Los resultados se muestran en la app tras el ajuste.

---
## 👥 Integrantes del equipo

**Helen Maureira - Francisco Provoste**
Estudiantes de Ciencia de Datos
Universidad Tecnológica Metropolitana (UTEM)
2025

---

## 📜 Licencia

Proyecto de uso académico.

# narrativa.md
# Cambios principales:
# - ARIMA como modelo principal
# - Descripción y recomendaciones actualizadas

# Narrativa del Proyecto: Predicción de Temperatura en Santiago

## Problema
Estimar la temperatura mínima y máxima diaria en Santiago de Chile para la toma de decisiones en actividades dependientes del clima.

## Objetivo
Construir una aplicación interactiva que permita:
- Explorar la serie histórica de temperaturas reales.
- Analizar la evolución mensual y diaria de la temperatura.
- Predecir la temperatura mínima y máxima para una fecha específica usando modelos avanzados.

## Datos
Dataset real (enero-noviembre 2025) con variables:
- momento (fecha y hora)
- tMin24Horas (temperatura mínima últimas 24h)
- tMax24Horas (temperatura máxima últimas 24h)

## Modelo
ARIMA (Modelo de series temporales, statsmodels):
- Predicción de temperaturas mínimas y máximas diarias.
- Métricas calculadas: MAE, MSE y R².

## Limitaciones y recomendaciones
- El modelo depende de la calidad y continuidad de los datos históricos.
- Recomendado: ampliar el dataset con más años, incluir variables adicionales (humedad, presión, eventos extremos) y comparar con otros modelos de series temporales (incluyendo LSTM y otros).
