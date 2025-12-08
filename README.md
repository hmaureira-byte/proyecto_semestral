# 🌤️ Predicción del clima en Santiago  
### Proyecto Semestral – Ciencia de Datos / Python + Streamlit + ML

Este proyecto desarrolla una aplicación interactiva para predecir el tiempo en Santiago de Chile utilizando modelos de *Machine Learning*.

La aplicación permite ingresar condiciones climáticas actuales y obtener predicciones en tiempo real mediante dos modelos:

- 🌳 **Random Forest**
- 📈 **Regresión Lineal**

---

## 🚀 Tecnologías utilizadas

- **Python 3.12**
- **Streamlit**
- **Scikit-learn**
- **Pandas / NumPy**
- **Joblib**
- **Git / GitHub**

---

## 🧠 Modelos entrenados

Los modelos se entrenaron con un dataset histórico del clima de Santiago y se incluyeron estas variables:

| Variable | Descripción |
|---------|-------------|
| temp | Temperatura (°C) |
| humedad | Humedad relativa (%) |
| viento | Velocidad del viento (m/s) |

Los modelos generados son:

- `random_forest_model.pkl`
- `linear_regression_model.pkl`

---

## 📂 Estructura del proyecto
📁 proyecto_clima_santiago
│── app.py
│── train.py
│── clima_santiago.csv
│── random_forest_model.pkl
│── linear_regression_model.pkl
│── README.md
└── venv/ (no subir)

## ▶️ Cómo ejecutar la aplicación

1. Crear entorno virtual:
python -m venv venv

2. Activarlo:
venv\Scripts\activate

3. Instalar dependencias:
pip install -r requirements.txt

4. Ejecutar la app:
streamlit run app.py

---

## 💡 Acerca del proyecto

El objetivo fue crear un sistema predictivo sencillo, rápido y accesible para estimar el clima en base a temperaturas registrada en el año 2024, demostrando:

- Procesamiento de datos
- Entrenamiento de modelos ML
- Integración con Streamlit
- Despliegue local
- Interpretación de resultados

---

## 👩‍💻 Autores

**Helen Maureira - Francisco Provoste**  
Estudiantes de Ciencia de Datos  
Universidad Tecnológica Metropolitana (UTEM)  
2025

---

## 📜 Licencia

Proyecto de uso académico.



