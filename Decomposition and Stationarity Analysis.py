import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

dates = pd.date_range('2025-01-01', periods=180, freq='D')

np.random.seed(10)

trend = np.linspace(100, 140, 180)

seasonal = 10 * np.sin(
    2 * np.pi * np.arange(180) / 30
)

noise = np.random.normal(0, 1.5, 180)

ts = pd.Series(
    trend + seasonal + noise,
    index=dates
)

result = seasonal_decompose(
    ts,
    model='additive',
    period=30
)

result.plot()
plt.show()

adf_result = adfuller(ts.dropna())

print('ADF statistic:', adf_result[0])
print('p-value:', adf_result[1])

diff_ts = ts.diff().dropna()

adf_diff = adfuller(diff_ts)

print('Differenced p-value:', adf_diff[1])
