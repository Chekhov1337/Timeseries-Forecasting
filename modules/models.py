from statsmodels.tsa.holtwinters import ExponentialSmoothing
from modules.stats import calc_time
import pmdarima as pm
import statsmodels as sm
import numpy as np


@calc_time
def holt_winters(train, start=109, end=120, seasonal_periods=12):
    predictions = list()

    for ts in train:
        model = ExponentialSmoothing(list(ts), seasonal=None, seasonal_periods=seasonal_periods).fit()
        predict = model.predict(start=start, end=end)
        predictions.append(predict)

    return predictions


def auto_arima(train, season=False, m=1, num_predictions=12):
    predictions = list()

    for ts in train:
        model = pm.auto_arima(ts, seasonal=season, stepwise=season, m=m, suppress_warnings=True,
                              error_action="ignore", max_order=None, trace=True)

        predictions.append(model.predict(num_predictions=num_predictions))

    return predictions


def ar_model(train, season=False, m=1, num_predictions=12):
    predictions = []
    for ts in train:
        ar = pm.auto_arima(ts, seasonal=season, stepwise=season, m=m, suppress_warnings=True,
                           error_action="ignore", max_order=None, trace=True, max_q=0, start_q=0, d=0)
        predictions.append(ar.predict(n_periods=num_predictions))

    return predictions


def ma_model(train, season=False, m=1, num_predictions=12):
    predictions = []
    for ts in train:
        ma = pm.auto_arima(ts, seasonal=season, stepwise=season, m=m, suppress_warnings=True,
                           error_action="ignore", max_order=None, trace=True, max_p=0, start_p=0, d=0)
        predictions.append(ma.predict(n_periods=num_predictions))

    return predictions


def arma_model(train, season=False, m=1):
    predictions = []
    for ts in train:
        arma = pm.auto_arima(ts, seasonal=season, stepwise=season, m=m, suppress_warnings=True,
                             error_action="ignore", max_order=None, trace=True, d=0)
        predictions.append(arma.predict(n_periods=12))

    return predictions

# def trend_model(train, period=12, num_predictions=12):
#     predictions = []
#     for ts in train:
#         decompose = sm.tsa.seasonal.seasonal_decompose(ts, period=period)
#         trend = decompose.trend
#         trend_new = np.ndarray(shape=120)
#         for i in range(108):
#             trend_new[i] = trend[i]
#         for i in range(12):
#             trend_new[108 + i] = None
