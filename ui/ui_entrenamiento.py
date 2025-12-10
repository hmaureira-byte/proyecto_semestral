import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ui.ui_modelo import get_lstm_model
import plotly.graph_objects as go
def main():
	def create_sequences(data, seq_length):
		xs, ys = [], []
		for i in range(len(data) - seq_length):
			x = data[i:(i+seq_length)]
			y = data[i+seq_length, :]
			xs.append(x)
			ys.append(y)
		return np.array(xs), np.array(ys)

	def train_lstm_model(pkl_path="data/temperatura_limpia_normalizada.pkl", target_col_name="ts", seq_length=24, epochs=30, lr=0.01):
		df = pd.read_pickle(pkl_path)
		# Usar solo tMin24Horas y tMax24Horas
		cols_temp = [col for col in ['tMin24Horas', 'tMax24Horas'] if col in df.columns]
		if len(cols_temp) < 2:
			st.warning("No se encontraron ambas columnas 'tMin24Horas' y 'tMax24Horas' en el dataset.")
			st.write(f"Columnas disponibles: {list(df.columns)}")
			return
		df_numerico = df[cols_temp]
		max_rows = 10000
		if len(df_numerico) > max_rows:
			st.warning(f"El dataset es muy grande. Se usará una muestra de {max_rows} filas para el entrenamiento.")
			df_numerico = df_numerico.sample(n=max_rows, random_state=42).sort_index()
		df_numerico = df_numerico.dropna()
		# Si el DataFrame queda vacío tras eliminar NaN, mostrar advertencia y no continuar
		if len(df_numerico) == 0:
			st.warning("No hay datos suficientes para entrenar el modelo. El DataFrame está vacío tras eliminar valores NaN. Verifica tu dataset o el preprocesamiento.")
			return
		# Verificar que hay suficientes filas para crear las secuencias
		if len(df_numerico) <= seq_length:
			st.error(f"No hay suficientes filas ({len(df_numerico)}) para crear secuencias de longitud {seq_length}. Reduce el parámetro 'seq_length' o usa más datos.")
			return
		if len(df_numerico) < seq_length + 1:
			st.error(f"La muestra seleccionada ({len(df_numerico)} filas) es demasiado pequeña para crear secuencias de longitud {seq_length}. Reduce el parámetro 'seq_length' o aumenta el tamaño de la muestra.")
			return
		data = df_numerico.values.astype(np.float32)
		X, y = create_sequences(data, seq_length)
		X_tensor = torch.from_numpy(X)
		y_tensor = torch.from_numpy(y)
		input_size = X.shape[2]
		output_size = y.shape[1]
		model = get_lstm_model(input_size, output_size=output_size)
		criterion = nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
		model.train()
		progress_bar = st.progress(0)
		status_text = st.empty()
		chart_area = st.empty()
		losses = []
		
		for epoch in range(epochs):
			optimizer.zero_grad()
			output = model(X_tensor)
			loss = criterion(output, y_tensor)
			loss.backward()
			optimizer.step()
			losses.append(loss.item())
			progress_bar.progress((epoch+1)/epochs)
			status_text.text(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
			fig = go.Figure()
			fig.add_trace(go.Scatter(x=list(range(1, len(losses)+1)), y=losses, mode='lines+markers', line=dict(color='blue')))
			fig.update_layout(title='Evolución del Loss durante el entrenamiento', xaxis_title='Epoch', yaxis_title='Loss')
			chart_area.plotly_chart(fig, use_container_width=True)
		torch.save(model.state_dict(), "data/lstm_temp_predictor.pth")
		st.success("Modelo LSTM entrenado y guardado.")
		model.eval()
		with torch.no_grad():
			y_pred = model(X_tensor).squeeze().numpy()
			y_true = y_tensor.squeeze().numpy()
			mae = mean_absolute_error(y_true, y_pred)
			mse = mean_squared_error(y_true, y_pred)
			r2 = r2_score(y_true, y_pred)
		st.markdown(f"**MAE:** {mae:.6f}")
		st.markdown(f"**MSE:** {mse:.6f}")
		st.markdown(f"**R²:** {r2:.6f}")

	st.title("Entrenamiento del modelo LSTM de temperatura")
	if st.button("Entrenar modelo LSTM"):
		train_lstm_model()
