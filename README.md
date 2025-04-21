# ARIMA Models Exploration (AirPassengers Dataset)

This project demonstrates time series forecasting using the ARIMA model, including differencing, stationarity checks, and backtesting.

We use the classic `AirPassengers` dataset, which records monthly totals of international airline passengers from 1949 to 1960.

---

## Mathematical Foundations

### 1. Stationarity

A time series is **stationary** if its statistical properties (mean, variance, autocorrelation) do not change over time. Many time series models assume stationarity. We test this with the **Augmented Dickey-Fuller (ADF)** test.

#### ADF Test Hypotheses

We test the model:

```
Δyₜ = α + βt + γyₜ₋₁ + δ₁Δyₜ₋₁ + ... + δₚΔyₜ₋ₚ + εₜ
```

Where:

- `Δyₜ = yₜ - yₜ₋₁`
- Null Hypothesis **H₀**: Series has a **unit root** (non-stationary)
- Alternative Hypothesis **H₁**: Series is stationary

We interpret the **p-value**:

- If `p < 0.05`, reject **H₀** → likely stationary  
- If `p >= 0.05`, fail to reject **H₀** → likely non-stationary

---

### 2. Differencing

Differencing transforms a non-stationary series into a stationary one by removing trends:

```
y′ₜ = yₜ - yₜ₋₁
```

This represents the **Integrated** part of ARIMA (the "I").

We compute it manually like so:

```python
def compute_difference(ts):
    ts_diff = []
    for i in range(1, len(ts)):
        ts_diff.append(ts[i] - ts[i-1])
    return pd.Series(ts_diff).dropna()
```

---

### 3. ARIMA Model

The **ARIMA** model consists of three components:

- **AR (AutoRegressive)**: Uses past values  
  ```
  yₜ = φ₁yₜ₋₁ + φ₂yₜ₋₂ + ... + φₚyₜ₋ₚ + εₜ
  ```

- **I (Integrated)**: Applies differencing to achieve stationarity  
  ```
  y′ₜ = yₜ - yₜ₋₁
  ```

- **MA (Moving Average)**: Uses past forecast errors  
  ```
  yₜ = εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θ_qεₜ₋q
  ```

Combined, the model is referred to as:

> **ARIMA(p, d, q)**  
> where `p` = AR order, `d` = differencing order, `q` = MA order

---

### 4. Forecast Recovery

Since we model on differenced data, we reverse differencing to return to the original scale:

```
ŷₜ = ŷ′ₜ + yₜ₋₁
```

In code, this is done via cumulative summation:

```python
np.cumsum(forecast_diff) + last_train_value
```

---

### 5. Model Evaluation

We evaluate model performance using **Root Mean Square Error (RMSE)**:

```
RMSE = sqrt( (1/n) * Σ (yᵢ - ŷᵢ)² )
```

Lower RMSE indicates better model performance on the test set.

---

## Results Summary

The **AR model** surprisingly performed the best on this dataset, both in terms of RMSE and MAE.

---

## Environment Activation

To activate the virtual environment on Windows:

```bash
 .\venv\Scripts\activate  
```
