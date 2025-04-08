"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

import pickle
import torch
from flwr.common.typing import NDArrays, Scalar, Metrics
from torch_cka import CKA
import numpy as np
from typing import List, Dict, Tuple
import csv
from sklearn.cluster import DBSCAN
import os
from sklearn.metrics import silhouette_score
from collections import Counter
import re
import pandas as pd
import ast


def save_parameters(parameters, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(parameters, f)
        
def calculate_gini_coefficient(income_list):
    # Sort the list in ascending order
    income_list = sorted(income_list)
    
    # Calculate the cumulative sum of the sorted list
    cumsum = sum(income_list)
    
    # Calculate the Gini coefficient using the formula
    numerator = 0
    for i, income in enumerate(income_list, 1):
        numerator += i * income
    gini = (2 * numerator) / (len(income_list) * cumsum) - (len(income_list) + 1) / len(income_list)
    
    return gini   


def set_parameters(net: torch.nn.Module, parameters: NDArrays) -> None:
    """Set parameters to a PyTorch network."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = dict({k: torch.Tensor(v) for k, v in params_dict})
    # ignore argument type because Dict keeps order in the supported python versions
    net.load_state_dict(state_dict, strict=True)  # type: ignore     
    
def compute_cka_similarity_matrix(data):
    """
    Compute the CKA similarity matrix for a list of model representations using torch-cka.
    
    Args:
    - data: A list of PyTorch tensors, each representing the flattened last layer parameters of a model.
    
    Returns:
    - A numpy array containing the CKA similarity matrix.
    """
    cka = CKA()  # Initialize the CKA calculator
    n = len(data)
    similarity_matrix = np.zeros((n, n))
    
    # Convert data list to a batch tensor if not already in one
    data_tensor = torch.stack(data) if not isinstance(data, torch.Tensor) else data
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Self-similarity
            else:
                # Compute CKA similarity between data[i] and data[j]
                sim = cka(data_tensor[i].unsqueeze(0), data_tensor[j].unsqueeze(0))
                similarity_matrix[i, j] = similarity_matrix[j, i] = sim.item()  # Assuming `sim` is a tensor
    
    return similarity_matrix


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """Compute weighted average.

    It is generic implementation that averages only over floats and ints
    and drops the other data types of the Metrics.
    """
    n_batches_list = [n_batches for n_batches, _ in metrics]
    n_batches_sum = sum(n_batches_list)
    metrics_lists: Dict[str, List[float]] = {}
    for number_of_batches, all_metrics_dict in metrics:
        #  Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            if isinstance(value, (float, int)):
                metrics_lists[single_metric] = []
        # Just one iteration needed to initialize the keywords
        break

    for number_of_batches, all_metrics_dict in metrics:
        # Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            # Add weighted metric
            if isinstance(value, (float, int)):
                metrics_lists[single_metric].append(float(number_of_batches * value))

    weighted_metrics: Dict[str, Scalar] = {}
    for metric_name, metric_values in metrics_lists.items():
        weighted_metrics[metric_name] = sum(metric_values) / n_batches_sum

    return weighted_metrics


def export_dict_to_csv(participation_dict, file_path):
    # Convert the dictionary to a list of dictionaries, where each dictionary is a row
    rows = [{"round": round_num, "participating_clients": ", ".join(clients)} for round_num, clients in participation_dict.items()]
    
    # Open the file and write the data using DictWriter
    with open(file_path, "w", newline="") as f:
        # Fieldnames are the keys of the dictionaries
        fieldnames = ["round", "participating_clients"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f'Successfully saved client participation data to {file_path}')
    
def generate_client_freq_figure(client_data):
    print("Generating frequency figure:")

    # Assuming client_data is your source dictionary
    server_rounds_tuple_lst = [(cid, client_dict["number_of_rounds"]) for cid, client_dict in client_data.items()]
    
    # Convert list of tuples to DataFrame
    df = pd.DataFrame(server_rounds_tuple_lst, columns=['cid', 'number_of_rounds'])

    # Sort the DataFrame by 'cid' in ascending order for sequential visualization
    df_sorted_by_cid = df.sort_values(by='cid', ascending=True)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_output_dir = hydra_cfg['runtime']['output_dir']
    os.chdir(hydra_output_dir)
    # Set the aesthetic style of the plots
    sns.set_theme(style="whitegrid")

    # Adjust the figure size to accommodate all client IDs
    fig, ax = plt.subplots(figsize=(20, 6))  # Adjust width as needed

    # Creating the bar chart
    sns.barplot(x='cid', y='number_of_rounds', data=df_sorted_by_cid, palette='viridis', ax=ax)

    # Improve the visualization
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Rotate the x-axis labels for better readability
    ax.set_title('Number of Rounds Each Client Participated In', fontsize=16)
    ax.set_xlabel('Client ID', fontsize=14)
    ax.set_ylabel('Number of Rounds', fontsize=14)

    plt.tight_layout() 
    # Save the figure
    plt.savefig('client_participation.png', dpi=300)  # Saves the figure to the current working directory
    plt.show()   
    
    
    
def get_class_distributions(self,):
    # Define the number of classes
    num_classes = 10
    epsilon = 1e-10
    
    # Calculate the class distribution in the training dataset
    class_counts = Counter()
    
    for _, label in self.trainloader.dataset:
        class_counts[label] += 1
    
        # Initialize counts for missing classes with zero
        for i in range(num_classes):
            if i not in class_counts:
                class_counts[i] = 0
    
    # Convert class counts to a dictionary and sort by class labels
    class_distribution = dict(sorted(class_counts.items()))
    
    # Transform counts to probabilities and replace zero values with epsilon
    total_count = sum(class_distribution.values())
    class_probabilities = {cls: (count if count > 0 else epsilon) / total_count for cls, count in class_distribution.items()}
        
def calculate_similarity(global_model_layer, client_model_layer) -> float:
        
        # Example: Cosine similarity
        dot_product = np.dot(global_model_layer.flatten(), client_model_layer.flatten())
        norm_global = np.linalg.norm(global_model_layer.flatten())
        norm_client = np.linalg.norm(client_model_layer.flatten())
        
        similarity = dot_product / (norm_global * norm_client)
        
        return similarity
    
    
def find_clusters(last_layers_with_cid: List[Tuple[str, np.ndarray]], eps=0.5, min_samples=2) -> Tuple[Dict[str, int], float]:
    # Extract weights from the tuples
    layer_weights_lst = [weights for _, weights in last_layers_with_cid]
    
    # Assuming each element in layer_weights_lst is a flat array of weights
    X = np.array(layer_weights_lst)

    # Initialize DBSCAN with the specified parameters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit DBSCAN to the data
    dbscan.fit(X)

    # Extract the cluster labels
    labels = dbscan.labels_

    # Check if more than one cluster was found to compute silhouette score
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(X, labels)
    else:
        silhouette_avg = -1  # Silhouette score is not meaningful if only one cluster or all points are noise

    # Create a dictionary mapping cids to their cluster labels
    cid_cluster_map = {cid: label for (cid, _), label in zip(last_layers_with_cid, labels)}

    return cid_cluster_map, silhouette_avg


def experiment_with_dbscan_and_print(last_layers_with_cid):
    """
    Experiment with different DBSCAN parameters and print the clustering results and silhouette scores.
    
    Parameters:
    - layer_weights_lst: List of flat arrays representing layer weights.
    """
    #layer_weights_lst = [weights for _, weights in last_layers_with_cid]
    
    # Define ranges for eps and min_samples to experiment with
    eps_values = np.linspace(0.1,0.6, 20)  # Example range for eps
    min_samples_values = range(2, 5)  # Example range for min_samples

    # Iterate over all combinations of eps and min_samples
    for eps in eps_values:
        for min_samples in min_samples_values:
            labels, silhouette_avg = find_clusters(last_layers_with_cid, eps=eps, min_samples=min_samples)
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise (-1)
            num_noise = list(labels).count(-1)
            
            if silhouette_avg > 0:
                print(f"eps: {eps:.2f}, min_samples: {min_samples}, "
                      f"Clusters: {num_clusters}, Noise points: {num_noise}, "
                      f"Silhouette Score: {silhouette_avg:.3f}")


def parse_log_file(log_file_path):
    # Initialize variables
    cluster_metrics = {}
    current_round = None
    cluster_ids = set()

    # Regular expressions to match lines
    round_start_pattern = re.compile(r'\[.*\] - Number of clients for round (\d+) is \d+')
    metrics_pattern = re.compile(r'\[.*\] - metrics for cluster\s+(\d+): (.*)')

    with open(log_file_path, 'r') as f:
        for line in f:
            # Check for round start
            round_match = round_start_pattern.match(line)
            if round_match:
                current_round = int(round_match.group(1))
                # Initialize the entry for the new round if not already present
                if current_round not in cluster_metrics:
                    cluster_metrics[current_round] = {}
                continue

            # Check for cluster metrics
            metrics_match = metrics_pattern.match(line)
            if metrics_match and current_round is not None:
                cluster_id = int(metrics_match.group(1))
                metrics_str = metrics_match.group(2)
                # Convert the metrics string to a dictionary using ast.literal_eval
                metrics_dict = ast.literal_eval(metrics_str)
                loss = metrics_dict.get('loss', None)

                # Store the loss for this cluster and round
                cluster_metrics[current_round][cluster_id] = loss
                cluster_ids.add(cluster_id)

    # Create a DataFrame from the cluster_metrics dictionary
    # Rows are rounds, columns are cluster IDs
    df = pd.DataFrame.from_dict(cluster_metrics, orient='index')

    # Sort the columns by cluster ID
    df = df.sort_index(axis=1)
    df.index.name = 'round'

    # Optionally, rename the columns to 'cluster_X'
    df.columns = [f'cluster_{int(col)}' for col in df.columns]

    # Reset index if you want 'round' as a column
    df = df.reset_index()

    return df
    
def save_metrics_to_csv(df, log_file_path, output_filename='cluster_losses.csv'):
    # Get the directory of the log file
    log_dir = os.path.dirname(os.path.abspath(log_file_path))
    # Construct the output file path
    output_file_path = os.path.join(log_dir, output_filename)

    # Check if the output file already exists
    if os.path.exists(output_file_path):
        print(f"Output file '{output_file_path}' already exists. Skipping save.")
        return
    else:
        # Save the DataFrame to a CSV file
        df.to_csv(output_file_path, index=False)
        print(f"Metrics saved to {output_file_path}")
    
def get_class_distributions_dataloader(dataloader):
    # Define the number of classes
    num_classes = 10
    epsilon = 1e-10
    
    # Calculate the class distribution in the training dataset
    class_counts = Counter()
    
    for _, label in dataloader.dataset:
        class_counts[label] += 1
    
        # Initialize counts for missing classes with zero
        for i in range(num_classes):
            if i not in class_counts:
                class_counts[i] = 0
                
                
    # Convert class counts to a dictionary and sort by class labels
    class_distribution = dict(sorted(class_counts.items()))
    
    return class_distribution

def constrained_output(output, min_value, max_value):
    """Apply min and max constraints to the PID output."""
    return max(min_value, min(max_value, output))


if __name__ == '__main__':
    # Usage
    log_file_path = 'outputs\\2024-09-14\\19-13-50\\main.log'  # Replace with your actual log file path

    # Check if the log file exists
    if not os.path.exists(log_file_path):
        print(f"Log file '{log_file_path}' not found.")
    else:
        # Parse the log file
        df = parse_log_file(log_file_path)
        # Save the metrics to a CSV file
        save_metrics_to_csv(df, log_file_path)
        # Optionally, print the DataFrame
        #print(df)
        
        
# niid_bench\utils.py