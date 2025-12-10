# Preprocesamiento de datos meteorológicos
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Unir los 11 datasets mensuales, limpiar y normalizar
import glob

# Buscar todos los archivos CSV de temperatura en la carpeta data
archivos = sorted(glob.glob("data/330019_2025*_Temperatura.csv"))

# Leer y unir todos los DataFrames
dfs = [pd.read_csv(archivo, sep=';') for archivo in archivos]
df = pd.concat(dfs, ignore_index=True)

# Limpiar: eliminar duplicados y rellenar nulos
df = df.drop_duplicates()
df = df.fillna(method="ffill").fillna(method="bfill")

# Filtrar solo las columnas de temperatura mínima y máxima 24h (y 'momento' si existe)
cols_temp = [col for col in ['tMin24Horas', 'tMax24Horas'] if col in df.columns]
if 'momento' in df.columns:
	df_filtrado = df[cols_temp + ['momento']]
else:
	df_filtrado = df[cols_temp]
# Normalizar solo las columnas de temperatura
scaler = MinMaxScaler()
df_filtrado[cols_temp] = scaler.fit_transform(df_filtrado[cols_temp])
# Guardar el DataFrame limpio y normalizado en un archivo .pkl
df_filtrado.to_pickle("data/temperatura_limpia_normalizada.pkl")