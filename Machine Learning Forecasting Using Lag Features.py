import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

dates = pd.date_range(
    '2025-01-01',
    periods=220,
    freq='D'
)

np.random.seed(60)

y = pd.Series(
    100
    + 0.1 * np.arange(220)
    + 5 * np.sin(
        2 * np.pi * np.arange(220) / 30
    )
    + np.random.normal(0, 1.5, 220),
    index=dates
)

df = pd.DataFrame({
    'y': y
})

df['lag1'] = df['y'].shift(1)

df['lag7'] = df['y'].shift(7)

df['lag14'] = df['y'].shift(14)

df = df.dropna()

split = int(len(df) * 0.8)

train = df.iloc[:split]

test = df.iloc[split:]

X_train = train[
    ['lag1', 'lag7', 'lag14']
]

y_train = train['y']

X_test = test[
    ['lag1', 'lag7', 'lag14']
]

y_test = test['y']

model = RandomForestRegressor(
    n_estimators=200,
    random_state=60
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred
    )
)

print('MAE:', mae)

print('RMSE:', rmse)

plt.figure(figsize=(10, 4))

plt.plot(
    y_test.index,
    y_test,
    label='Actual'
)

plt.plot(
    y_test.index,
    pred,
    label='Predicted'
)

plt.legend()

plt.show()
