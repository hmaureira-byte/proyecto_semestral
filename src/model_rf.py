from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def train_rf(df, save_path="../models/rf_model.pkl"):
    """
    Entrena un Random Forest y lo guarda.
    """
    X = df[["humidity", "wind_speed", "pressure"]]
    y = df["temperature"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, save_path)

    return model
