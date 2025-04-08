"""Fed Cosine"""

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
import pdb
from collections import defaultdict

from niid_bench.utils import calculate_similarity,find_clusters, experiment_with_dbscan_and_print

def set_parameters(self, parameters):
    """Set the local model parameters using given ones."""
    params_dict = zip(self.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    self.load_state_dict(state_dict, strict=True)
    
class FedLocalLoss(FedAvg):
    def __init__(self, fraction_fit: float = 1.0, *args, **kwargs):
        self.num_rounds = kwargs.pop('num_rounds', 30)  # Default to 30 if not provided
        super().__init__(*args, **kwargs)
        self.fraction_fit = fraction_fit
        self.last_global_weights = None
        self.last_local_loss = {}
        self.client_prob_by_cid = {}
       
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        
        config = {}
        
        config['server_round'] = server_round
        
        # Get all clients
        clients = list(client_manager.all().values())
        num_clients = len(clients)
        
        num_participating_clients = 12
        # Calculate the number of clients for the current round
       
        # Prepare the fit instructions for the selected clients
        
        if server_round == 1:
            fit_ins = FitIns(parameters, config)
            client_instructions = [(client, fit_ins) for client in clients]
            logging.info(f'Number of clients for round {server_round} is {num_clients}')
            
        else:
            # For subsequent rounds, select clients based on their local loss
            sorted_clients = sorted(
                clients,
                key=lambda client: self.last_local_loss.get(client.cid, float('-inf')),  # Sort by loss value or use a large negative number if not found
                reverse=True  # Highest loss first
            )
            
            # Select the top clients based on the number of clients to participate
            selected_clients = sorted_clients[:num_participating_clients]
            
            # Create FitIns for selected clients
            fit_ins = FitIns(parameters, config)
            client_instructions = [(client, fit_ins) for client in selected_clients]
            
            logging.info(f'Selected clients for round {server_round}: {[client.cid for client in selected_clients]}')
                
        return client_instructions
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        parameters_aggregated, _ = super().aggregate_fit(server_round,results,failures)

       
        logging.info('Started updating last_local_loss dict')
        # update dict with last local loss info
        for (client, fit_res) in results:
            cid = client.cid
            self.last_local_loss[cid] = fit_res.metrics['local_loss']
            
        logging.info('Finished updating elf.last_local_loss dict')
        
        logging.info(f'printing updated local losses: {self.last_local_loss}')
        
        return parameters_aggregated, {}
    
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
        
    
