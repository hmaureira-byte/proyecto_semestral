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

import streamlit as st

from ui.ui_home import main as pantalla_principal
from ui.ui_analitica_web import main as analisis_datos
from ui.ui_modelo import main as prediccion
from ui.ui_entrenamiento import main as entrenar

# Configuración de la página
st.set_page_config(
    page_title="🌦️ Clima Santiago",
    page_icon="🌡️",
    layout="wide"
)

# Título principal
st.title("🌦️ Sistema de Predicción de Temperatura en Santiago")

# Menú lateral
seccion = st.sidebar.selectbox(
    "Navegación",
    ["Inicio", "Análisis de datos", "Predicción", "Entrenar"]
)

# Navegación
if seccion == "Inicio":
    pantalla_principal()
elif seccion == "Análisis de datos":
    analisis_datos()
elif seccion == "Predicción":
    prediccion()
elif seccion == "Entrenar":
    entrenar()
