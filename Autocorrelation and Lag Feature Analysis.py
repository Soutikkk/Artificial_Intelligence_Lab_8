import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf

dates = pd.date_range('2025-01-01', periods=120, freq='D')

np.random.seed(21)

ts = pd.Series(
    50 + np.cumsum(
        np.random.normal(0, 1, 120)
    ),
    index=dates
)

lag_1 = ts.shift(1)

lag_7 = ts.shift(7)

df = pd.DataFrame({
    'y': ts,
    'lag1': lag_1,
    'lag7': lag_7
}).dropna()

print(df.head())

print(
    'Lag-1 correlation:',
    df['y'].corr(df['lag1'])
)

plot_acf(ts, lags=30)

plt.show()
