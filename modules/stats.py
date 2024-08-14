import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error
from tabulate import tabulate
from datetime import datetime as dt


def plot(original_values, predictions, test_values, label, start_n=1, save_to_file=False):
    plt.figure(figsize=(10, 6))
    for ts_orig, ts_pred, ts_test in zip(original_values, predictions, test_values):
        time_steps = list(range(len(ts_orig)))
        future_time_steps = list(range(len(ts_orig), len(ts_orig) + len(ts_pred)))
        if predictions is not None:
            plt.plot(time_steps, ts_orig, label='Original Values', color='blue')
            plt.plot(future_time_steps, ts_pred, label='Predicted Values', color='red', linestyle='--')
            plt.plot(future_time_steps, ts_test, label='Test Values', color='green', linestyle=':')
            plt.title(f'Series {start_n} - {label} Forecast vs Actual')
        else:
            plt.plot(time_steps + future_time_steps, np.concatenate((ts_orig, ts_test)),
                     label='Original Values', color='blue')
            plt.title(f'Time Series № {start_n} - {label}')

        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.legend()

        start_n += 1

        if save_to_file:
            script_dir = os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, f'figures\\{label}')
            sample_file_name = f'Series {start_n} - {label}'

            os.makedirs(results_dir, exist_ok=True)
            plt.savefig(os.path.join(results_dir, sample_file_name))

        plt.show()


def metrics(predictions, test, print_tables=False):
    metrics = {
        'RMSE': [],
        'R^2': [],
        'MAPE': [],
        'SMAPE': []
    }

    # Calculate metrics for each series
    for test, pred in zip(test, predictions):
        actual_values = test.to_numpy()
        predicted_values = pred

        # Ensure the predicted values and actual values have the same length
        min_length = min(len(actual_values), len(predicted_values))
        actual_values = actual_values[:min_length]
        predicted_values = predicted_values[:min_length]

        # Calculate RMSE
        rmse = root_mean_squared_error(actual_values, predicted_values)
        metrics['RMSE'].append(rmse)

        # Calculate R^2
        r2 = r2_score(actual_values, predicted_values)
        metrics['R^2'].append(r2)

        # Calculate MAPE
        mape = mean_absolute_percentage_error(actual_values, predicted_values)
        metrics['MAPE'].append(mape)

        # Calculate SMAPE
        smape = np.mean(
            np.abs(predicted_values - actual_values) /
            ((np.abs(predicted_values) + np.abs(actual_values)) / 2)
        )
        metrics['SMAPE'].append(smape)

    mean_rmse = np.mean(metrics['RMSE'])
    mean_r2 = np.mean(metrics['R^2'])
    mean_mape = np.mean(metrics['MAPE'])
    mean_smape = np.mean(metrics['SMAPE'])

    # Create a new dictionary for the average metrics
    average_metrics = {
        'RMSE': [mean_rmse],
        'R^2': [mean_r2],
        'MAPE': [mean_mape],
        'SMAPE': [mean_smape]
    }

    if print_tables:
        print('Average Metrics:')
        print(tabulate(average_metrics, headers=['RMSE', 'R^2', 'MAPE', 'SMAPE'], tablefmt="rounded_outline",
                       numalign='left'))
        print('Metrics for each timeseries:')
        print(tabulate(metrics, headers=['RMSE', 'R^2', 'MAPE', 'SMAPE'], tablefmt="rounded_outline", numalign='left'))

    average_metrics_df = pd.DataFrame(average_metrics)
    metrics_df = pd.DataFrame(metrics)

    return metrics_df, average_metrics_df


def calc_time(model):
    def wrapper(*args, **kwargs):
        start = dt.now()
        res = model(*args, **kwargs)
        end = dt.now()
        print(f'Forecasting time: {end - start}')
        return res

    return wrapper
