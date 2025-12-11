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
cols_12h = [col for col in ['tMin12Horas', 'tMax12Horas'] if col in df.columns]
if 'momento' in df.columns:
    df_filtrado = df[cols_temp + cols_12h + ['momento']]
    df_filtrado['momento'] = pd.to_datetime(df_filtrado['momento'])
    df_filtrado = df_filtrado.sort_values('momento').reset_index(drop=True)
else:
    df_filtrado = df[cols_temp + cols_12h]

# Rellenar valores faltantes de 24h con los de 12h si existen
df_filtrado['tMin24Horas'] = df_filtrado['tMin24Horas'].combine_first(df_filtrado['tMin12Horas']) if 'tMin24Horas' in df_filtrado.columns and 'tMin12Horas' in df_filtrado.columns else df_filtrado['tMin24Horas']
df_filtrado['tMax24Horas'] = df_filtrado['tMax24Horas'].combine_first(df_filtrado['tMax12Horas']) if 'tMax24Horas' in df_filtrado.columns and 'tMax12Horas' in df_filtrado.columns else df_filtrado['tMax24Horas']

# Eliminar filas con NaN en las columnas de temperatura principales
df_filtrado = df_filtrado.dropna(subset=cols_temp)

# Guardar el DataFrame limpio (sin normalizar) en un archivo .pkl
df_filtrado.to_pickle("data/temperatura_limpia.pkl")