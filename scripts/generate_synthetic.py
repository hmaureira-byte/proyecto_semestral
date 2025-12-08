# scripts/generate_synthetic.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
start = datetime(2024,1,1)
hours = 24*180  # 6 months
rows=[]
for i in range(hours):
    dt = start + timedelta(hours=i)
    temp = 10 + 8*np.sin(2*np.pi*(i/24)/30) + np.random.normal(0,2)
    rows.append({
        "datetime": dt,
        "temp": temp,
        "humidity": np.random.uniform(40,80),
        "pressure": np.random.uniform(1005,1025),
        "clouds": np.random.uniform(0,100),
        "wind_speed": np.random.uniform(0,6),
        "weather_main": np.random.choice(["Clear","Clouds","Rain","Fog"])
    })
df = pd.DataFrame(rows)
df.to_csv("data/santiago_historical.csv", index=False)
print("CSV sintético creado en data/santiago_historical.csv")
