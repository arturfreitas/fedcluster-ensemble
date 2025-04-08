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
from typing import OrderedDict

import json
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, FitIns

def set_parameters(self, parameters):
    """Set the local model parameters using given ones."""
    params_dict = zip(self.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    self.load_state_dict(state_dict, strict=True)
    
class FedMedian(FedAvg):
    def __init__(self, fraction_fit: float = 1.0, *args, **kwargs):
        self.num_rounds = kwargs.pop('num_rounds', 100)  # Default to 100 if not provided
        super().__init__(*args, **kwargs)
        self.fraction_fit = fraction_fit
        self.last_parameters_aggregated = None
        self.last_metrics_aggregated = None
        self.median_similarity = 0
        logging.info("Strategy Init complete")

    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        
        logging.info(f"Configuring fit for round {server_round}")
        # Get all clients
        all_clients = list(client_manager.all().values())
        
        config = {}
        
        config["server_round"] = server_round
        config["selection_thrshold"] = self.median_similarity # zero for first round
        
        logging.info(f'Config threshold for round {server_round} is {config["selection_thrshold"]}')

        # Convert parameters to weights
        self.last_parameters_aggregated = parameters_to_ndarrays(parameters)
        
        # Prepare the fit instructions for the selected clients
        fit_ins = FitIns(parameters, config)
        
        # For the first round, include all clients
        client_instructions = [(client, fit_ins) for client in all_clients ]
            
        return client_instructions
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        selected_client_ids = [int(client.cid) for client, fit_res in results if len(fit_res.parameters.tensors) > 0]
        selected_client_ids.sort()
        
        n_clients = len(selected_client_ids)
        
        logging.info(f" Total clients participating on round {server_round}: {n_clients}")
        
        if n_clients > 0:
            logging.info(f"Clients participating on round {server_round}: {selected_client_ids}")
        
            # generating similarity list of participating clients
            self.similarity_list = [ res.metrics["similarity"] for client, res in results if int(client.cid) in selected_client_ids]
        
            logging.info(f"List of similarities:")
        
            for client, res in results:
                if int(client.cid) in selected_client_ids:
                    logging.info(f"{client.cid}: {res.metrics['similarity']}")
            
            self.median_similarity = statistics.median(self.similarity_list)
        
            logging.info(f"Calculated median similarity is: {self.median_similarity}")
            
            # Filter the results to include only the selected clients (non empty results)
            results_filtered = [(client, fit_res) for client, fit_res in results if int(client.cid) in selected_client_ids]
            
            parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results_filtered, failures)
            
            self.last_parameters_aggregated = parameters_aggregated
            self.last_metrics_aggregated = metrics_aggregated 
        else:
            # it means none of the clients met the threshhold
            logging.info(f"None of the clients met the criteria on round {server_round}. Using previous model for next round")
            parameters_aggregated, metrics_aggregated = self.last_parameters_aggregated, self.last_metrics_aggregated

        return parameters_aggregated, metrics_aggregated
    
    def parameters_to_weights(self, parameters: Parameters):
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
        
    
