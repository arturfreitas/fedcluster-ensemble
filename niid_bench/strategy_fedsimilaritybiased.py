"""FedNova and SCAFFOLD strategies."""

from functools import reduce
from logging import WARNING

import numpy as np
import pdb
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.common.logger import log
from flwr.common.typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statistics
import hydra
import datetime
import os
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
#from utils import gini_coefficient
from flwr.server import ClientManager
import random
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from niid_bench.models import CNNMnist
from typing import OrderedDict
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import json
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, FitIns

from niid_bench.utils import calculate_similarity

def set_parameters(self, parameters):
    """Set the local model parameters using given ones."""
    params_dict = zip(self.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    self.load_state_dict(state_dict, strict=True)
    
class FedSimilarityBiased(FedAvg):
    def __init__(self, fraction_fit: float = 1.0, *args, **kwargs):
        self.num_rounds = kwargs.pop('num_rounds', 30)  # Default to 30 if not provided
        super().__init__(*args, **kwargs)
        self.fraction_fit = fraction_fit
        self.last_global_weights = None
        self.similarity_by_cid = {}
        self.client_participation_count = {}
       
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        
        # Get all clients
        clients = list(client_manager.all().values())
        
        fraction_fit = 0.7
        
        num_clients = int(fraction_fit * len(clients)) 
        
        config = {}
        config['server_round'] = server_round
        # Calculate alfa (the decrement per round)
        
        # Calculate the number of clients for the current round
        
        if server_round == 1:
            # Sample all clients on the first round
            selected_clients = clients
        else:
            
            # Create a list of client cids
            cids_lst = [client.cid for client in clients]
            
            # Use similarity to compute probabilities for client selection
            similarities = np.array([self.similarity_by_cid.get(cid, 1.0) for cid in cids_lst])
            
            # Shift similarities to a non-negative range [0, 2] (since cosine similarity can be [-1, 1])
            shifted_similarities = similarities + 1.0
            
            probabilities = np.exp(shifted_similarities) / np.sum(np.exp(shifted_similarities))
            
            # Apply softmax to calculate probabilities for client selection
            exp_inverted_similarities = np.exp(shifted_similarities)
            probabilities = exp_inverted_similarities / np.sum(exp_inverted_similarities)
            
            # Perform weighted random sampling based on the probabilities, using cids
            selected_cids = np.random.choice(cids_lst, num_clients, replace=False, p=probabilities)
            
            # Sort the selected cids (they are strings, so we sort by numeric value)
            selected_cids = sorted(selected_cids, key=lambda x: int(x))  # Sorting based on numeric values
            
            # Print the selected cids (sorted numerically)
            print(f'Selected Client IDs for round {server_round}: {selected_cids}')
            
            # Retrieve the actual client objects corresponding to the selected cids
            selected_clients = [client for client in clients if client.cid in selected_cids]

        # Prepare the fit instructions for the selected clients
        fit_ins = FitIns(parameters, config)
        client_instructions = [(client, fit_ins) for client in selected_clients]
        
        return client_instructions
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        self.clients_with_similarities = []

        # Perform the aggregation (e.g., FedAvg) to update the global model
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        
        logging.info(f"printing loss aggregated for round {server_round}")
        logging.info(metrics_aggregated['loss'])
        
        self.last_global_weights = parameters_to_ndarrays(parameters_aggregated)
        
        for client, fit_res in results:
            cid = client.cid
            # Request the current model from the client
            client_params = fit_res.parameters
            client_weights = parameters_to_ndarrays(client_params)
            
            # Calculate similarity (e.g., cosine similarity) between the last layers
            similarity = calculate_similarity(self.last_global_weights[-1], client_weights[-1])
            self.similarity_by_cid[cid] = similarity
            self.client_participation_count[cid] = self.client_participation_count.get(cid,0) + 1
            
        for cid, sim in self.similarity_by_cid.items():
            logging.info(f'Client {client.cid}: {sim}')
        
        return parameters_aggregated, metrics_aggregated
   

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
        
    
