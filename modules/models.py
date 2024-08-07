from statsmodels.tsa.holtwinters import ExponentialSmoothing


def holt_winters(train, start=109, end=120, seasonal_periods=12):
    predicts = list()

    for i in range(len(train)):
        model = ExponentialSmoothing(train[i], seasonal=None, seasonal_periods=seasonal_periods).fit()
        predict = model.predict(strart=start, end=end)
        predicts.append(predict)

    return predicts
