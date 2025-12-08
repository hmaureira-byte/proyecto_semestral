import joblib
import pandas as pd

def predict_temperature(humidity, wind_speed, pressure,
                        arima_path="../models/arima_model.pkl",
                        rf_path="../models/rf_model.pkl"):

    arima = joblib.load(arima_path)
    rf = joblib.load(rf_path)

    # Random Forest (inputs directos)
    rf_pred = rf.predict([[humidity, wind_speed, pressure]])[0]

    # ARIMA → usamos último valor + ruido
    arima_pred = float(arima.forecast(steps=1))

    return {
        "random_forest": rf_pred,
        "arima": arima_pred
    }
