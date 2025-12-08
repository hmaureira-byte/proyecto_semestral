import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pickle
import os

def train_arima(df):
    # Asegurar que el índice es datetime
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")

    # Usamos solo la temperatura como serie temporal
    series = df["temp"]

    # Entrenamos modelo (puedes ajustar el orden si quieres mejorar)
    model = ARIMA(series, order=(2,1,2))
    model_fit = model.fit()

    # Guardar modelo
    os.makedirs("models", exist_ok=True)
    with open("models/arima_model.pkl", "wb") as f:
        pickle.dump(model_fit, f)

    print("Modelo ARIMA guardado correctamente.")
    return model_fit


def predict_arima(df, hours=24):
    # Cargar modelo
    with open("models/arima_model.pkl", "rb") as f:
        model_fit = pickle.load(f)

    # Última fecha real del dataset
    df["datetime"] = pd.to_datetime(df["datetime"])
    last_date = df["datetime"].max()

    # Generar fechas futuras reales
    future_index = pd.date_range(
        start=last_date + pd.Timedelta(hours=1),
        periods=hours,
        freq="H"
    )

    # Predicción
    forecast = model_fit.forecast(steps=hours)
    forecast = pd.DataFrame({
        "datetime": future_index,
        "pred_temp": forecast.values
    })

    return forecast
if __name__ == "__main__":
    # Prueba rápida
    df = pd.read_csv("data/santiago_historical.csv")
    model = train_arima(df)
    forecast = predict_arima(df, hours=24)
    print(forecast)
    print("Predicción ARIMA para las próximas 24 horas:")
    print(forecast)

