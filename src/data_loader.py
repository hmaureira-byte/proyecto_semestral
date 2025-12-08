import pandas as pd
import os

def load_weather_data(path="data/santiago_historical.csv"):

    """
    Carga el archivo CSV, valida columnas y retorna el DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ No se encontró el archivo en {path}")

    df = pd.read_csv(path, parse_dates=["datetime"])

    expected_columns = {"datetime", "temperature", "humidity", "wind_speed", "pressure"}
    if not expected_columns.issubset(df.columns):
        raise ValueError("❌ El CSV no tiene las columnas correctas.")

    print("✅ Archivo CSV cargado correctamente.")
    return df