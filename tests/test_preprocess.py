# tests/test_preprocess.py
import pandas as pd
from src.preprocess import basic_clean, fill_missing_weather

def test_basic_clean():
    df = pd.DataFrame({"dt":[1650000000], "temp":[15]})
    df2 = basic_clean(df)
    assert "datetime" in df2.columns

def test_fill_missing_weather():
    df = pd.DataFrame({"datetime":[pd.Timestamp("2024-01-01")], "temp":[15], "humidity":[None]})
    df2 = fill_missing_weather(df)
    assert df2["humidity"].isnull().sum() == 0
