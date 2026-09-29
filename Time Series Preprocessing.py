import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dates = pd.date_range('2025-01-01', periods=120, freq='D')

np.random.seed(7)

values = 100 + np.cumsum(
    np.random.normal(0, 1, len(dates))
)

ts = pd.Series(values, index=dates, name='sales')

ts.iloc[[10, 35, 70]] = np.nan

print('Missing values:', ts.isna().sum())

ts_filled = ts.interpolate(method='time')

weekly = ts_filled.resample('W').mean()

rolling_7 = ts_filled.rolling(7).mean()

plt.figure(figsize=(10, 4))
plt.plot(ts_filled, label='Original / filled')
plt.plot(rolling_7, label='7-day rolling mean')

plt.legend()
plt.title('Preprocessing and Smoothing')
plt.show()
