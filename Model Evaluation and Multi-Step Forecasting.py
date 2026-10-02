import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

np.random.seed(90)

dates = pd.date_range(
    '2025-01-01',
    periods=240,
    freq='D'
)

y = pd.Series(
    50
    + 0.12 * np.arange(240)
    + 4 * np.sin(
        2 * np.pi * np.arange(240) / 30
    )
    + np.random.normal(0, 1.2, 240),
    index=dates
)

df = pd.DataFrame({
    'y': y
})

for lag in [1, 7, 14, 30]:
    df[f'lag{lag}'] = df['y'].shift(lag)

df = df.dropna()

split = int(len(df) * 0.8)

train = df.iloc[:split]

test = df.iloc[split:]

features = [
    'lag1',
    'lag7',
    'lag14',
    'lag30'
]

model = RandomForestRegressor(
    n_estimators=250,
    random_state=90
)

model.fit(
    train[features],
    train['y']
)

pred = model.predict(
    test[features]
)

naive = test['lag1']

mae_model = mean_absolute_error(
    test['y'],
    pred
)

rmse_model = np.sqrt(
    mean_squared_error(
        test['y'],
        pred
    )
)

mae_naive = mean_absolute_error(
    test['y'],
    naive
)

rmse_naive = np.sqrt(
    mean_squared_error(
        test['y'],
        naive
    )
)

print('Model MAE:', mae_model)

print('Model RMSE:', rmse_model)

print('Naive MAE:', mae_naive)

print('Naive RMSE:', rmse_naive)
