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
from flower.server.strategy import FedAvg
from flower.server.client_manager import ClientManager
from flower.server.client_proxy import ClientProxy
from flower.common import Parameters, FitIns, Weights

def set_parameters(self, parameters):
    """Set the local model parameters using given ones."""
    params_dict = zip(self.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    self.load_state_dict(state_dict, strict=True)
    
class FedCosine(FedAvg):
    def __init__(self, fraction_fit: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fraction_fit = fraction_fit
        self.last_global_weights = None

    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        
        # Get all clients
        clients = list(client_manager.all().values())

        if server_round == 1:
            # Sample all clients on the first round
            selected_clients = clients
        else:
            # Convert parameters to weights (assuming model is using weights)
            global_weights = self.parameters_to_weights(parameters)
            self.last_global_weights = global_weights
            
            # Calculate similarity of client models with the global model
            client_similarities = []
            for client in clients:
                # Request the current model from the client
                client_params = client.get_parameters() 
                client_weights = self.parameters_to_weights(client_params)
                
                # Calculate similarity (e.g., cosine similarity) between the last layers
                similarity = self.calculate_similarity(global_weights[-1], client_weights[-1])
                client_similarities.append((client, similarity))
            
            # Sort clients by similarity
            client_similarities.sort(key=lambda x: x[1], reverse=True)
            
            logging.info(f'Client similarities for round number {server_round}')
            
            # Print client similarities in the desired format
            for client, similarity in client_similarities:
                print(f"{client.cid} : {similarity}")
            
            # Select a fraction of clients based on similarity
            num_clients = int(self.fraction_fit * len(clients))
            selected_clients = [client for client, _ in client_similarities[:num_clients]]
        
        # Prepare the fit instructions for the selected clients
        fit_ins = FitIns(parameters, {})
        client_instructions = [(client, fit_ins) for client in selected_clients]
        
        return client_instructions

    def parameters_to_weights(self, parameters: Parameters) -> Weights:
        # Convert Parameters to Weights
        return parameters.tensors  # assuming parameters are already in a tensor format

    def calculate_similarity(self, global_layer, client_layer) -> float:
        
        # Example: Cosine similarity
        dot_product = np.dot(global_layer.flatten(), client_layer.flatten())
        norm_global = np.linalg.norm(global_layer.flatten())
        norm_client = np.linalg.norm(client_layer.flatten())
        
        similarity = dot_product / (norm_global * norm_client)
        
        return similarity
   

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
        
    
