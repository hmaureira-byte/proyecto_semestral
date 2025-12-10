def main():
	import streamlit as st
	import pandas as pd
	import matplotlib.pyplot as plt
	df = pd.read_pickle("data/temperatura_limpia_normalizada.pkl")
	if not pd.api.types.is_datetime64_any_dtype(df["momento"]):
		df["momento"] = pd.to_datetime(df["momento"])
	st.title("Análisis Exploratorio de Temperatura")
	# Verificar columnas necesarias
	if "tMin24Horas" not in df.columns or "tMax24Horas" not in df.columns:
		st.warning("Las columnas 'tMin24Horas' y/o 'tMax24Horas' no existen en el DataFrame. Verifica el nombre de las columnas en tu archivo de datos.")
		st.write(f"Columnas disponibles: {list(df.columns)}")
		return

	# Desnormalizar usando los valores originales (asumiendo que tienes los min/max originales)
	# Si tienes los valores originales, reemplaza estos por los correctos:
	tmin_min, tmin_max = 0, 40
	tmax_min, tmax_max = 0, 45
	df['tMin24Horas_desnorm'] = df['tMin24Horas'] * (tmin_max - tmin_min) + tmin_min
	df['tMax24Horas_desnorm'] = df['tMax24Horas'] * (tmax_max - tmax_min) + tmax_min

	import plotly.graph_objects as go
	st.subheader("Serie temporal de temperaturas mínimas y máximas (últimas 24h)")
	fig1 = go.Figure()
	fig1.add_trace(go.Scatter(x=df["momento"], y=df["tMin24Horas_desnorm"], mode='lines', name='Mínima 24h', line=dict(color='blue')))
	fig1.add_trace(go.Scatter(x=df["momento"], y=df["tMax24Horas_desnorm"], mode='lines', name='Máxima 24h', line=dict(color='red')))
	fig1.update_layout(xaxis_title="Fecha y hora", yaxis_title="Temperatura (°C)")
	st.plotly_chart(fig1, use_container_width=True)

	st.subheader("Histograma de temperaturas mínimas y máximas (últimas 24h)")
	fig2 = go.Figure()
	fig2.add_trace(go.Histogram(x=df["tMin24Horas_desnorm"], name='Mínima 24h', marker_color='blue', opacity=0.6))
	fig2.add_trace(go.Histogram(x=df["tMax24Horas_desnorm"], name='Máxima 24h', marker_color='red', opacity=0.6))
	fig2.update_layout(xaxis_title="Temperatura (°C)", yaxis_title="Frecuencia", barmode='overlay')
	st.plotly_chart(fig2, use_container_width=True)

	st.subheader("Media mensual de temperaturas mínimas y máximas (últimas 24h)")
	df["mes"] = df["momento"].dt.to_period("M")
	medias_min = df.groupby("mes")["tMin24Horas_desnorm"].mean()
	medias_max = df.groupby("mes")["tMax24Horas_desnorm"].mean()
	fig3 = go.Figure()
	fig3.add_trace(go.Scatter(x=medias_min.index.astype(str), y=medias_min.values, mode='lines+markers', name='Mínima 24h', line=dict(color='blue')))
	fig3.add_trace(go.Scatter(x=medias_max.index.astype(str), y=medias_max.values, mode='lines+markers', name='Máxima 24h', line=dict(color='red')))
	fig3.update_layout(xaxis_title="Mes", yaxis_title="Temperatura media (°C)")
	st.plotly_chart(fig3, use_container_width=True)
