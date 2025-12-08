from data_loader import load_weather_data
from preprocessing import clean_weather_df, create_features
from model_arima import train_arima
from model_rf import train_rf
from utils import banner
import os
import pickle

def main():
    banner("Cargando datos...")
    df = load_weather_data()

    banner("Limpieza...")
    df = clean_weather_df(df)

    banner("Creando features...")
    df = create_features(df)

    # Crear carpeta models
    os.makedirs("models", exist_ok=True)

    banner("Entrenando ARIMA...")
    arima_model = train_arima(df)     # ← GUARDAMOS EL MODELO

    banner("Entrenando Random Forest...")
    rf_model = train_rf(df)           # ← GUARDAMOS EL MODELO

    # GUARDADO DE MODELOS
    with open("models/arima_model.pkl", "wb") as f:
        pickle.dump(arima_model, f)

    with open("models/rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    banner("✔ ENTRENAMIENTO COMPLETADO Y MODELOS GUARDADOS")

if __name__ == "__main__":
    main()
