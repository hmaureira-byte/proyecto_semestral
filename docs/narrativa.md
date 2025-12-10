# Narrativa del Proyecto: Predicción de Temperatura en Santiago

## Problema
Estimar la temperatura mínima y máxima diaria en Santiago de Chile para mejorar la planificación urbana, la gestión energética y la toma de decisiones en actividades dependientes del clima.

## Objetivo
Construir una aplicación interactiva que permita:
- Explorar la serie histórica de temperaturas reales.
- Analizar la evolución mensual y diaria de la temperatura.
- Predecir la temperatura mínima y máxima para una fecha específica usando modelos avanzados.

## Datos
Dataset real (enero-noviembre 2025) con variables:
- momento (fecha y hora)
- tMin24Horas (temperatura mínima últimas 24h)
- tMax24Horas (temperatura máxima últimas 24h)

## Modelo
LSTM (Red Neuronal Recurrente, PyTorch):
- Predicción multivariada de temperaturas mínimas y máximas.
- Métricas calculadas: MAE, MSE y R².

## Limitaciones y recomendaciones
- El modelo depende de la calidad y continuidad de los datos históricos.
- Recomendado: ampliar el dataset con más años, incluir variables adicionales (humedad, presión, eventos extremos) y comparar con otros modelos de series temporales.
