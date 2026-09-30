import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing

dates = pd.date_range('2025-01-01', periods=150, freq='D')

np.random.seed(30)

ts = pd.Series(
    80 + 0.15 * np.arange(150) +
    np.random.normal(0, 2, 150),
    index=dates
)

train = ts.iloc[:-20]

test = ts.iloc[-20:]

ma_value = train.tail(7).mean()

ma_forecast = pd.Series(
    ma_value,
    index=test.index
)

model = ExponentialSmoothing(
    train,
    trend='add',
    seasonal=None
).fit()

holt_forecast = model.forecast(len(test))

plt.figure(figsize=(10, 4))

plt.plot(train, label='Train')

plt.plot(test, label='Test')

plt.plot(
    ma_forecast,
    label='Moving average forecast'
)

plt.plot(
    holt_forecast,
    label='Holt forecast'
)

plt.legend()

plt.show()
