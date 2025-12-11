import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
def main():
	def train_arima_model(pkl_path="data/temperatura_limpia.pkl"):
		df = pd.read_pickle(pkl_path)
		cols_temp = [col for col in ['tMin24Horas', 'tMax24Horas'] if col in df.columns]
		if len(cols_temp) < 2:
			st.warning("No se encontraron ambas columnas 'tMin24Horas' y 'tMax24Horas' en el dataset.")
			st.write(f"Columnas disponibles: {list(df.columns)}")
			return
		df = df.dropna()

		st.subheader("Ajuste ARIMA para temperatura mínima")
		model_min = ARIMA(df['tMin24Horas'], order=(2,1,2)).fit()
		st.success("Modelo ARIMA ajustado para temperatura mínima.")
		# Métricas para temperatura mínima
		y_true_min = df['tMin24Horas']
		y_pred_min = model_min.fittedvalues
		mae_min = mean_absolute_error(y_true_min, y_pred_min)
		mse_min = mean_squared_error(y_true_min, y_pred_min)
		r2_min = r2_score(y_true_min, y_pred_min)
		st.markdown(f"**MAE (mínima):** {mae_min:.2f}")
		st.markdown(f"**MSE (mínima):** {mse_min:.2f}")
		st.markdown(f"**R² (mínima):** {r2_min:.4f}")
		# Gráfico de ajuste mínimo con fechas
		fechas = df['momento'] if 'momento' in df.columns else df.index
		if not isinstance(fechas, pd.Series):
			fechas = pd.Series(fechas)
		fechas = pd.to_datetime(fechas)
		etiquetas = fechas.dt.strftime('%b-%d')
		fig_min = go.Figure()
		fig_min.add_trace(go.Scatter(x=etiquetas, y=y_true_min, mode='lines', name='Real'))
		fig_min.add_trace(go.Scatter(x=etiquetas, y=y_pred_min, mode='lines', name='ARIMA'))
		fig_min.update_layout(title='Ajuste ARIMA - Temperatura Mínima', xaxis_title='Fecha', yaxis_title='Temperatura (°C)')
		st.plotly_chart(fig_min, use_container_width=True)

		st.subheader("Ajuste ARIMA para temperatura máxima")
		model_max = ARIMA(df['tMax24Horas'], order=(2,1,2)).fit()
		st.success("Modelo ARIMA ajustado para temperatura máxima.")
		# Métricas para temperatura máxima
		y_true_max = df['tMax24Horas']
		y_pred_max = model_max.fittedvalues
		mae_max = mean_absolute_error(y_true_max, y_pred_max)
		mse_max = mean_squared_error(y_true_max, y_pred_max)
		r2_max = r2_score(y_true_max, y_pred_max)
		st.markdown(f"**MAE (máxima):** {mae_max:.2f}")
		st.markdown(f"**MSE (máxima):** {mse_max:.2f}")
		st.markdown(f"**R² (máxima):** {r2_max:.4f}")
		# Gráfico de ajuste máxima con fechas
		fechas = df['momento'] if 'momento' in df.columns else df.index
		if not isinstance(fechas, pd.Series):
			fechas = pd.Series(fechas)
		fechas = pd.to_datetime(fechas)
		etiquetas = fechas.dt.strftime('%b-%d')
		fig_max = go.Figure()
		fig_max.add_trace(go.Scatter(x=etiquetas, y=y_true_max, mode='lines', name='Real'))
		fig_max.add_trace(go.Scatter(x=etiquetas, y=y_pred_max, mode='lines', name='ARIMA'))
		fig_max.update_layout(title='Ajuste ARIMA - Temperatura Máxima', xaxis_title='Fecha', yaxis_title='Temperatura (°C)')
		st.plotly_chart(fig_max, use_container_width=True)

		# Guardar modelos
		import joblib
		joblib.dump(model_min, "data/arima_min_model.pkl")
		joblib.dump(model_max, "data/arima_max_model.pkl")
		st.info("Modelos ARIMA guardados.")

	st.title("Entrenamiento de modelos ARIMA de temperatura")
	if st.button("Entrenar modelos ARIMA"):
		train_arima_model()
