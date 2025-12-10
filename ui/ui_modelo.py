# Importaciones
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Definición del modelo LSTM para uso en Streamlit
class LSTMTempPredictor(nn.Module):
	def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
		super(LSTMTempPredictor, self).__init__()
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		out, _ = self.lstm(x)
		out = self.fc(out[:, -1, :])
		return out

# Función para instanciar el modelo
def get_lstm_model(input_size, hidden_size=64, num_layers=2, output_size=1):
	return LSTMTempPredictor(input_size, hidden_size, num_layers, output_size)

# Función para cargar el modelo entrenado
def cargar_modelo_lstm(input_size, path="data/lstm_temp_predictor.pth"):
	model = get_lstm_model(input_size, output_size=2)
	model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
	model.eval()
	return model


def mostrar():
	st.title("Predicción de Temperatura (LSTM)")

	st.subheader("Selecciona los datos para predecir:")
	meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre"]
	mes_idx = st.selectbox("Mes", list(range(1, 12)), format_func=lambda x: meses[x-1])
	dia = st.number_input("Día", min_value=2, max_value=30, value=2)

	df = pd.read_pickle("data/temperatura_limpia_normalizada.pkl")
	df['momento'] = pd.to_datetime(df['momento'])
	# Usar solo tMin24Horas y tMax24Horas como variables
	cols_temp = [col for col in ['tMin24Horas', 'tMax24Horas'] if col in df.columns]
	if len(cols_temp) < 2:
		st.warning("No se encontraron ambas columnas 'tMin24Horas' y 'tMax24Horas' en el dataset.")
		st.write(f"Columnas disponibles: {list(df.columns)}")
		return
	input_size = len(cols_temp)
	model = cargar_modelo_lstm(input_size)

	fecha_str = f"2025-{mes_idx:02d}-{dia:02d}"
	idx = df[df['momento'].dt.strftime('%Y-%m-%d') == fecha_str].index
	if len(idx) == 0 or idx[0] < 24:
		st.warning("No hay suficientes datos para esa fecha o la fecha no existe en el dataset.")
		return

	secuencia_df = df.iloc[idx[0]-24:idx[0]][cols_temp]
	secuencia_np = secuencia_df.values.astype(np.float32).reshape(1, 24, len(cols_temp))
	secuencia_tensor = torch.from_numpy(secuencia_np)

	if st.button("Predecir temperatura"):
		with torch.no_grad():
			pred = model(secuencia_tensor).numpy()
		# Desnormalizar usando los valores reales del dataset
		tmin_min, tmin_max = -0.3, 29.4
		tmax_min, tmax_max = 4.1, 36.1
		temp_min = pred[0][0] * (tmin_max - tmin_min) + tmin_min
		temp_max = pred[0][1] * (tmax_max - tmax_min) + tmax_min
		st.subheader(f"Predicción estimada:")
		st.write(f"Temperatura mínima (24h): **{temp_min:.2f} °C**")
		st.write(f"Temperatura máxima (24h): **{temp_max:.2f} °C**")

# Función main para integración con app.py
def main():
	mostrar()