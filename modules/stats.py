import matplotlib.pyplot as plt
import os
import numpy as np


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
