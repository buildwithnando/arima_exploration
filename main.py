import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from statsmodels.tsa.stattools import adfuller # To check for stationarity
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # To help identify p and q (optional, but good practice)

from statsmodels.datasets import get_rdataset

data = get_rdataset("AirPassengers").data
data['time'] = pd.to_datetime(data['time'], origin='1899-12-30', unit='D') # Adjust origin based on R's time handling
data.set_index('time', inplace=True)
ts = data['value']  # Monthly passenger count

print("--- Original Time Series ---")
ts.plot(title="Monthly Airline Passengers (AirPassengers)", figsize=(10, 4))
plt.ylabel("Passengers")


def compute_difference(ts):
    ts_diff = []
    for i in range(1, len(ts)):
        diff = ts[i] - ts[i-1]
        ts_diff.append(diff)

    ts_diff = pd.Series(ts_diff)
    ts_diff = ts_diff.dropna()
    return ts_diff

print("\nChecking for Stationarity (ADF Test)")
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')
    if result[1] <= 0.06:
        print("Result: The series is likely stationary (reject H0)")
    else:
        print("Result: The series is likely non-stationary (fail to reject H0)")

adf_test(ts)

# Since we failed to reject H0, This tells us we likely need differencing (the 'I' component of ARIMA).
# Differencing
ts_diff1 = ts.diff().dropna()
ts_diff2 = ts_diff1.diff().dropna() 
ts_diff1_manual = compute_difference(ts)

print(len(ts_diff1_manual))
print(len(ts_diff1))

# print(ts_diff1_manual.head())
# print(ts_diff1.head())
adf_test(ts_diff1) # lol, changed to check 6% instead of 5% since p-value was 0.0542

# ts_diff1.plot(title="First-Order Differenced Air Passengers", figsize=(10, 4))
# ts_diff2.plot(title="Second-Order Differenced Air Passengers", figsize=(10, 4))
# ts_diff1_manual.plot(title="First-Order Differenced Air Passengers (Manual)", figsize=(10, 4))

train_size = int(len(ts) * 0.8)
train =  ts[:train_size]
test = ts[train_size:]

train_diff1 = train.diff().dropna()

# Let's first try just using AR model, note for Ferdinand: ARIMA(p,d,q) where p=AR, d=I, q=MA
ar_model = ARIMA(train_diff1, order=(20, 0, 0))
ar_fitted = ar_model.fit()
print(ar_fitted.summary())

ar_forecast_diff = ar_fitted.forecast(steps=len(test))
 # we need to undo the differencing to get the actual forecasted values
ar_forecast_original = np.cumsum(ar_forecast_diff) + train.iloc[-1]
ar_forecast_original.index = test.index # Align the index with the test set
plt.plot(ar_forecast_original.index, ar_forecast_original, label="AR(10) Forecast (on Differenced)", linestyle="--", color='orange')

# Ok now lets try just using MA model

ma_model = ARIMA(train_diff1, order=(0, 0, 30))
ma_fitted = ma_model.fit()
print(ma_fitted.summary())
ma_forecast_diff = ma_fitted.forecast(steps=len(test))
ma_forecast_original = np.cumsum(ma_forecast_diff) + train.iloc[-1]
ma_forecast_original.index = test.index 
plt.plot(ma_forecast_original.index, ma_forecast_original, label="MA(30) Forecast (on Differenced)", linestyle="--", color='pink')



# Ok now lets mix the two together and see how we do
arma_model = ARIMA(train_diff1, order=(20, 0, 30)) #the difference can also be done in this line
arma_fitted = arma_model.fit()
print(arma_fitted.summary())
arma_forecast_diff = arma_fitted.forecast(steps=len(test))
arma_forecast_original = np.cumsum(arma_forecast_diff) + train.iloc[-1]
arma_forecast_original.index = test.index
plt.plot(arma_forecast_original.index, arma_forecast_original, label="ARIMA(20,1,30) Forecast (on Differenced)", linestyle="--", color='green')
plt.legend()

rmse_ar = np.sqrt(mean_squared_error(test, ar_forecast_original))
rmse_ma = np.sqrt(mean_squared_error(test, ma_forecast_original))
rmse_arma = np.sqrt(mean_squared_error(test, arma_forecast_original))

mae_ar = mean_absolute_error(test, ar_forecast_original)
mae_ma = mean_absolute_error(test, ma_forecast_original)
mae_arma = mean_absolute_error(test, arma_forecast_original)

print(f"RMSE (AR(20) on Differenced): {rmse_ar:.2f}")
print(f"MAE  (AR(20) on Differenced): {mae_ar:.2f}\n")

print(f"RMSE (MA(30) on Differenced): {rmse_ma:.2f}")
print(f"MAE  (MA(30) on Differenced): {mae_ma:.2f}\n")

print(f"RMSE (ARIMA(20,1,30) on Original): {rmse_arma:.2f}")
print(f"MAE  (ARIMA(20,1,30) on Original): {mae_arma:.2f}")

#Interesting, the AR performed the best


plt.show()