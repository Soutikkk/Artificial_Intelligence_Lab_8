import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

np.random.seed(50)

dates = pd.date_range(
    '2024-01-01',
    periods=180,
    freq='D'
)

y = pd.Series(
    200 + np.cumsum(
        np.random.normal(0, 1.2, 180)
    ),
    index=dates
)

train = y.iloc[:-30]

test = y.iloc[-30:]

model = ARIMA(
    train,
    order=(1, 1, 1)
).fit()

forecast = model.forecast(
    steps=len(test)
)

plt.figure(figsize=(10, 4))

plt.plot(train, label='Train')

plt.plot(test, label='Actual test')

plt.plot(
    forecast,
    label='ARIMA forecast'
)

plt.legend()

plt.title('ARIMA Forecast')

plt.show()
