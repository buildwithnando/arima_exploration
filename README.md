
# ARIMA Models Exploration (AirPassengers Dataset)

This project demonstrates time series forecasting using the ARIMA model from first principles, including differencing, stationarity checks, and backtesting.

We use the classic `AirPassengers` dataset which records monthly totals of international airline passengers from 1949 to 1960.

---

## Mathematical Foundations

### 1. Stationarity

A time series is *stationary* if its properties (mean, variance, autocorrelation) do not change over time. Many time series models assume stationarity. To test this, we use the **Augmented Dickey-Fuller (ADF)** test.

#### ADF Test Hypotheses

We test the model:

$$
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \delta_1 \Delta y_{t-1} + \dots + \delta_p \Delta y_{t-p} + \epsilon_t
$$

Where:

- $ \Delta y_t = y_t - y_{t-1} $
- Null Hypothesis $ H_0 $: Series has a **unit root** (non-stationary)
- Alternative $H_1 $: Series is stationary

We check the **p-value** to determine stationarity:
- If \( p < 0.05 \), reject $ H_0 $: Series is likely stationary
- If \( p >= 0.05 \), fail to reject $ H_0 $: Series is non-stationary

---

### 2. Differencing

Differencing transforms a non-stationary series into a stationary one by removing trends:

$$
y'_t = y_t - y_{t-1}
$$

This is the **integrated** part of ARIMA (the 'I'). We manually compute this with a loop:

```python
def compute_difference(ts):
    ts_diff = []
    for i in range(1, len(ts)):
        ts_diff.append(ts[i] - ts[i-1])
    return pd.Series(ts_diff).dropna()
```

---

### 3. ARIMA Model

The **ARIMA** model is composed of:

- **AR (AutoRegressive)**: uses past values  
  $$
  y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \epsilon_t
  $$

- **I (Integrated)**: differencing to make data stationary  
  $$
  y'_t = y_t - y_{t-1}
  $$

- **MA (Moving Average)**: uses past forecast errors  
  $$
  y_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}
  $$

Combined ARIMA(p,d,q):

$$
\text{ARIMA}(p,d,q): \text{AutoRegressive order } p,\ \text{Differencing order } d,\ \text{Moving Average order } q
$$

---

### 4. Forecast Recovery

Since we model on differenced data, we reverse differencing to get back to the original scale:

$$
\hat{y}_t = \hat{y}'_t + y_{t-1}
$$

We use cumulative sum:

```python
np.cumsum(forecast_diff) + last_train_value
```

---

### 5. Model Evaluation

We evaluate using **Root Mean Square Error (RMSE)**:

$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
$$

Lower RMSE indicates better model fit on the test set.

---

## Results Summary

**AR model** surprisingly performed the best on this dataset, RMSE wise and MAE wise. 



## Env Activation 

```
 .\venv\Scripts\activate  
```
