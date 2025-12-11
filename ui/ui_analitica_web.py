def main():
	import streamlit as st
	import pandas as pd
	import matplotlib.pyplot as plt
	# Cambiar el nombre del archivo si corresponde
	df = pd.read_pickle("data/temperatura_limpia.pkl")
	if not pd.api.types.is_datetime64_any_dtype(df["momento"]):
		df["momento"] = pd.to_datetime(df["momento"])
	st.title("Análisis Exploratorio de Temperatura")
	# Verificar columnas necesarias
	if "tMin24Horas" not in df.columns or "tMax24Horas" not in df.columns:
		st.warning("Las columnas 'tMin24Horas' y/o 'tMax24Horas' no existen en el DataFrame. Verifica el nombre de las columnas en tu archivo de datos.")
		st.write(f"Columnas disponibles: {list(df.columns)}")
		return

	# Usar directamente los valores reales
	import plotly.graph_objects as go
	st.subheader("Serie temporal de temperaturas mínimas y máximas")
	fig1 = go.Figure()
	fig1.add_trace(go.Scatter(x=df["momento"], y=df["tMin24Horas"], mode='lines', name='Mínima', line=dict(color='blue')))
	fig1.add_trace(go.Scatter(x=df["momento"], y=df["tMax24Horas"], mode='lines', name='Máxima', line=dict(color='red')))
	fig1.update_layout(xaxis_title="Fecha y hora", yaxis_title="Temperatura (°C)")
	st.plotly_chart(fig1, use_container_width=True)

	st.subheader("Histograma de temperaturas mínimas y máximas")
	fig2 = go.Figure()
	fig2.add_trace(go.Histogram(x=df["tMin24Horas"], name='Mínima', marker_color='blue', opacity=0.6))
	fig2.add_trace(go.Histogram(x=df["tMax24Horas"], name='Máxima', marker_color='red', opacity=0.6))
	fig2.update_layout(xaxis_title="Temperatura (°C)", yaxis_title="Frecuencia", barmode='overlay')
	st.plotly_chart(fig2, use_container_width=True)

	st.subheader("Media mensual de temperaturas mínimas y máximas")
	df["mes"] = df["momento"].dt.to_period("M")
	medias_min = df.groupby("mes")["tMin24Horas"].mean()
	medias_max = df.groupby("mes")["tMax24Horas"].mean()
	fig3 = go.Figure()
	fig3.add_trace(go.Scatter(x=medias_min.index.astype(str), y=medias_min.values, mode='lines+markers', name='Mínima', line=dict(color='blue')))
	fig3.add_trace(go.Scatter(x=medias_max.index.astype(str), y=medias_max.values, mode='lines+markers', name='Máxima', line=dict(color='red')))
	fig3.update_layout(xaxis_title="Mes", yaxis_title="Temperatura media (°C)")
	st.plotly_chart(fig3, use_container_width=True)
