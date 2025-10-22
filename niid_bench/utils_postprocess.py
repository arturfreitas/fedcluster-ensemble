import matplotlib.pyplot as plt
import numpy as np


def filter_experiment(exp, criteria):
    for key, value in criteria.items():
        # Check if the value is a list, meaning we want to match any of the values in the list
        attr_value = getattr(exp, key)
        if isinstance(value, list):
            if attr_value not in value:
                return False
        else:
            if attr_value != value:
                return False
    return True

def filter_experiments(experiments, criteria):
    return [exp for exp in experiments if filter_experiment(exp, criteria)]

def plot_centralized_metrics(experiments, title="Experiment Results", metric = 'centralized_accuracy', label_attr = 'ID'):
    exp_strategy_cnt = {}
    
    # Create a new figure with the specified size and DPI
    figsize=(12, 8)
    plt.figure(figsize=figsize, dpi=100)
    for exp in experiments:
        if exp.centralized_metrics is not None:
             # Generate a sequence of increasing values starting from 1
            rounds = range(1, len(exp.centralized_metrics) + 1)
            label = getattr(exp, label_attr, 'Unknown')
            plt.plot(rounds, exp.centralized_metrics[metric], label=f"{label}")
            #exp_strategy_cnt[exp.strategy] = exp_strategy_cnt.get(exp.strategy, 0) + 1

    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(metric)
    plt.legend()
    plt.show()
    
    
    
def plot_distributed_metrics(experiments, title="Experiment Results", metric = 'aggregated_test_accuracy', label_attr = 'strategy'):
    for exp in experiments:
        if exp.centralized_metrics is not None:
             # Generate a sequence of increasing values starting from 1
            rounds = range(1, len(exp.centralized_metrics) )
            label = getattr(exp, label_attr, 'Unknown')
            plt.plot(rounds, exp.distributed_metrics[metric], label=f"{label_attr} = {label}")

    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(metric)
    plt.legend()
    plt.show()
    
def plot_fourier_metric(experiments, metric='centralized_accuracy', title="Fourier Transform of Metric", label_attr='strategy'):
    """Plots the Fourier Transform of a selected metric from the experiment results.

    Parameters
    ----------
    experiments : list
        List of experiment objects containing metrics.
    metric : str
        The metric to be analyzed (default is 'centralized_accuracy').
    title : str
        The title of the plot.
    label_attr : str
        The attribute used for labeling different experiments.
    """
    plt.figure(figsize=(12, 8), dpi=100)

    for exp in experiments:
        if exp.centralized_metrics is not None:
            # Extract the metric values and compute the Fourier Transform
            data = np.array(exp.centralized_metrics[metric])
            fourier_transform = np.fft.fft(data)
            freqs = np.fft.fftfreq(len(data))

            # Only use positive frequencies and their magnitudes
            positive_freqs = freqs[:len(freqs) // 2]
            positive_magnitudes = np.abs(fourier_transform)[:len(freqs) // 2]

            # Get the label for the plot
            label = getattr(exp, label_attr, 'Unknown')

            # Plot the magnitude spectrum of the positive frequencies
            plt.plot(positive_freqs, positive_magnitudes, label=f"{label_attr} = {label}")

    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()