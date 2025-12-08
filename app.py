# app.py (versión WOW)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import io

# ---------------------------
# Página y estilos
# ---------------------------
st.set_page_config(page_title="🌦️ Clima Santiago", layout="wide", initial_sidebar_state="expanded")

# CSS para apariencia
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%); }
    .big-title {font-size:34px; font-weight:700; color:#0f1724;}
    .card { background: #ffffff; border-radius: 12px; padding: 18px; box-shadow: 0 4px 18px rgba(15,23,36,0.06); }
    .kpi { font-size:28px; font-weight:700; color:#0b69ff; }
    .small { color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Helpers
# ---------------------------
@st.cache_resource
def load_models():
    with open("models/arima_model.pkl", "rb") as f:
        arima_model = pickle.load(f)
    with open("models/rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    return arima_model, rf_model

def get_last_row(df):
    return df.sort_values("datetime").iloc[-1]

def make_forecast_df_arima(df, hours=24):
    # Usa la función predict_arima si existe
    try:
        from model_arima import predict_arima
        return predict_arima(df, hours=hours)
    except Exception:
        # fallback: usar model directamente si necesario
        with open("models/arima_model.pkl","rb") as f:
            model = pickle.load(f)
        df["datetime"] = pd.to_datetime(df["datetime"])
        last_date = df["datetime"].max()
        future_index = pd.date_range(start=last_date+pd.Timedelta(hours=1), periods=hours, freq="H")
        forecast = model.forecast(steps=hours)
        return pd.DataFrame({"datetime": future_index, "pred_temp": forecast})

def make_forecast_df_rf(df, hours=24, rf_model=None):
    # Simulación simple: usa la última fila repetida (fast approach)
    last = get_last_row(df)
    rows = []
    base_features = [last["temperature"], last["humidity"], last["wind_speed"]]
    for i in range(hours):
        X = np.array(base_features).reshape(1, -1)
        pred = float(rf_model.predict(X)[0])
        # Optionally shift base_features[0] = pred to simulate iterative forecast:
        base_features[0] = pred
        rows.append({"datetime": last["datetime"] + timedelta(hours=i+1), "pred_temp": pred})
    return pd.DataFrame(rows)

def df_to_csv_bytes(df):
    b = io.BytesIO()
    df.to_csv(b, index=False)
    b.seek(0)
    return b

# ---------------------------
# Cargar modelos y datos
# ---------------------------
arima_model, rf_model = load_models()

st.title("🌦️ Predicción del Clima en Santiago — *WOW Dashboard*")
st.markdown("Aplicación con modelos ARIMA + Random Forest — diseño profesional para entrega universitaria.")

# Sidebar: opciones
with st.sidebar:
    st.header("Configuración")
    HOURS = st.slider("Horas a predecir (ARIMA / RF)", min_value=6, max_value=72, value=24, step=6)
    show_shap = st.checkbox("Mostrar SHAP (si está disponible)", value=False)
    st.write("---")
    st.markdown("**Datos**")
    uploaded = st.file_uploader("Sube CSV histórico (opcional)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded, parse_dates=["datetime"])
    else:
        df = pd.read_csv("data/santiago_historical.csv", parse_dates=["datetime"])

# Top KPIs (current)
df = df.sort_values("datetime")
last = get_last_row(df)
k1, k2, k3 = st.columns([1.5,1,1])

with k1:
    st.markdown('<div class="card"><div class="small">Última lectura</div><div class="kpi">{:.2f} °C</div><div class="small">{} (registro más reciente)</div></div>'.format(last["temperature"], last["datetime"]), unsafe_allow_html=True)
with k2:
    st.markdown('<div class="card"><div class="small">Humedad</div><div class="kpi">{:.0f} %</div></div>'.format(last["humidity"]), unsafe_allow_html=True)
with k3:
    st.markdown('<div class="card"><div class="small">Viento</div><div class="kpi">{:.2f} m/s</div></div>'.format(last["wind_speed"]), unsafe_allow_html=True)

st.markdown("---")

# Main layout: Historical + controls
left, right = st.columns([2,1])

with left:
    st.subheader("📈 Serie histórica — Temperatura")
    fig = px.line(df.tail(500), x="datetime", y="temperature", title="Temperatura (últimas 500 horas)")
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 🔮 Predicciones comparadas")
    colA, colB = st.columns(2)

    # ARIMA forecast
    with colA:
        if st.button("Generar Predicción ARIMA"):
            with st.spinner("Generando ARIMA..."):
                arima_df = make_forecast_df_arima(df, hours=HOURS)
            st.success("ARIMA generado.")
            st.dataframe(arima_df.head(HOURS))
            fig_a = px.line(arima_df, x="datetime", y="pred_temp", title="ARIMA - Pronóstico")
            st.plotly_chart(fig_a, use_container_width=True)

    # RF forecast
    with colB:
        st.subheader("Random Forest - Predicción iterativa (basada en último registro)")
        if st.button("Generar Predicción RF"):
            with st.spinner("Generando RF..."):
                rf_df = make_forecast_df_rf(df, hours=HOURS, rf_model=rf_model)
            st.success("Random Forest generado.")
            st.dataframe(rf_df.head(HOURS))
            fig_r = px.line(rf_df, x="datetime", y="pred_temp", title="Random Forest - Pronóstico")
            st.plotly_chart(fig_r, use_container_width=True)

    # If both generated, show comparison
    if ("arima_df" in globals()) and ("rf_df" in globals()):
        cmp = pd.merge(arima_df, rf_df, on="datetime", how="outer", suffixes=("_arima","_rf")).dropna()
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=cmp["datetime"], y=cmp["pred_temp_arima"], mode="lines+markers", name="ARIMA"))
        fig_cmp.add_trace(go.Scatter(x=cmp["datetime"], y=cmp["pred_temp_rf"], mode="lines+markers", name="Random Forest"))
        fig_cmp.update_layout(title="Comparación ARIMA vs Random Forest", xaxis_title="Fecha", yaxis_title="Temperatura (°C)")
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("---")
    st.subheader("📥 Exportar predicción (CSV)")
    if "arima_df" in globals():
        csv_bytes = df_to_csv_bytes(arima_df)
        st.download_button("Descargar ARIMA CSV", data=csv_bytes, file_name="arima_forecast.csv", mime="text/csv")
    if "rf_df" in globals():
        csv_bytes2 = df_to_csv_bytes(rf_df)
        st.download_button("Descargar RF CSV", data=csv_bytes2, file_name="rf_forecast.csv", mime="text/csv")

with right:
    st.subheader("🧭 Predicción puntual (Random Forest)")
    st.markdown("Ingresa condiciones (Nota: la presión se ignora para este RF entrenado con 3 features).")
    col1, col2 = st.columns(2)
    with col1:
        temp_in = st.number_input("Temperatura actual (°C)", min_value=-30.0, max_value=50.0, value=float(last["temperature"]))
        hum_in = st.number_input("Humedad (%)", min_value=0.0, max_value=100.0, value=float(last["humidity"]))
    with col2:
        wind_in = st.number_input("Velocidad viento (m/s)", min_value=0.0, max_value=50.0, value=float(last["wind_speed"]))
        press_in = st.number_input("Presión (hPa) (IGNORADA)", min_value=800, max_value=1100, value=int(last.get("pressure",1013)))

    if st.button("Predecir temperatura (RF)"):
        X = np.array([[temp_in, hum_in, wind_in]])  # 3 features
        try:
            pred = float(rf_model.predict(X)[0])
            st.metric("🌡️ Temperatura estimada (°C)", f"{pred:.2f}")
            # small interpretation
            delta = pred - temp_in
            if delta > 1:
                st.success(f"Subida estimada ≈ {delta:.2f} °C — condiciones más cálidas")
            elif delta < -1:
                st.warning(f"Bajada estimada ≈ {delta:.2f} °C — condiciones más frías")
            else:
                st.info("Cambio pequeño esperado (≈ {:.2f} °C).".format(delta))
        except Exception as e:
            st.error(f"Error al predecir: {e}")

    st.markdown("---")
    st.subheader("ℹ️ Sobre esta app")
    st.markdown("""
    - Modelos: **ARIMA** (series temporales) y **Random Forest** (regresión).
    - ARIMA predice usando la serie completa de temperatura.
    - Random Forest usa las features: `temperature`, `humidity`, `wind_speed`.
    - Para mayor precisión puedes re-entrenar RF incluyendo `pressure`.
    """)

# SHAP (opcional)
if show_shap:
    st.markdown("---")
    st.subheader("🔍 Explicabilidad (SHAP)")
    try:
        import shap
        sample_X = np.array([[float(last["temperature"]), float(last["humidity"]), float(last["wind_speed"])]])
        explainer = shap.TreeExplainer(rf_model)
        shap_vals = explainer.shap_values(sample_X)
        st.text("SHAP values (sample): " + str(shap_vals))
    except Exception as e:
        st.write("SHAP no disponible o muy pesado en este entorno:", e)

st.markdown("---")
st.caption("Proyecto de Helen & Francisco — Predicción Clima (Santiago).")
# Fin de app.py