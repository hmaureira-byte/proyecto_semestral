
import streamlit as st
import pandas as pd
import numpy as np
import joblib


def mostrar():
	st.title("Predicción de Temperatura (ARIMA)")

	st.subheader("Selecciona los datos para predecir:")
	meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre"]
	mes_idx = st.selectbox("Mes", list(range(1, 12)), format_func=lambda x: meses[x-1])
	dia = st.number_input("Día", min_value=2, max_value=30, value=2)

	df = pd.read_pickle("data/temperatura_limpia.pkl")
	df['momento'] = pd.to_datetime(df['momento'])
	fecha_str = f"2025-{mes_idx:02d}-{dia:02d}"
	idx = df[df['momento'].dt.strftime('%Y-%m-%d') == fecha_str].index
	if len(idx) == 0:
		st.warning("No hay datos para esa fecha o la fecha no existe en el dataset.")
		return

	# Cargar modelos ARIMA entrenados
	try:
		model_min = joblib.load("data/arima_min_model.pkl")
		model_max = joblib.load("data/arima_max_model.pkl")
	except Exception as e:
		st.error(f"No se encontraron los modelos ARIMA entrenados. Entrena primero en la sección correspondiente. Error: {e}")
		return

	# Predecir para la fecha seleccionada
	pred_idx = idx[0]
	pred_min = model_min.predict(start=pred_idx, end=pred_idx)
	pred_max = model_max.predict(start=pred_idx, end=pred_idx)

	st.subheader(f"Predicción estimada:")
	st.write(f"Temperatura mínima: **{pred_min.iloc[0]:.2f} °C**")
	st.write(f"Temperatura máxima: **{pred_max.iloc[0]:.2f} °C**")

# Función main para integración con app.py
def main():
	mostrar()