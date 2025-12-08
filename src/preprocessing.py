import pandas as pd

def clean_weather_df(df):
    """
    Realiza limpieza básica del dataset.
    """
    df = df.copy()
    
    # Eliminar duplicados
    df = df.drop_duplicates()

    # Rellenar posibles nulos (por seguridad)
    df = df.fillna(method="ffill").fillna(method="bfill")

    return df


def create_features(df):
    """
    Crea variables adicionales útiles para modelos predictivos.
    """
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["dayofweek"] = df["datetime"].dt.dayofweek
    return df
def preprocess_weather_data(df):
    """
    Función completa de preprocesamiento del dataset de clima.
    """
    df = clean_weather_df(df)
    df = create_features(df)
    return df
