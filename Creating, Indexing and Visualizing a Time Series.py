import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

dates = pd.date_range('2025-01-01', periods=180, freq='D')

trend = np.linspace(50, 90, len(dates))

seasonal = 8 * np.sin(
    2 * np.pi * np.arange(len(dates)) / 30
)

noise = np.random.normal(0, 2, len(dates))

values = trend + seasonal + noise

ts = pd.Series(values, index=dates, name='value')

print(ts.head())
print(ts.describe())

plt.figure(figsize=(10, 4))
plt.plot(ts)
plt.title('Synthetic Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()