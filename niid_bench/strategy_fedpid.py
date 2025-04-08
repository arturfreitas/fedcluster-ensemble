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
import csv
from niid_bench.utils import calculate_similarity,find_clusters, experiment_with_dbscan_and_print,constrained_output
from simple_pid import PID
import hydra
from hydra.core.hydra_config import HydraConfig


def set_parameters(self, parameters):
    """Set the local model parameters using given ones."""
    params_dict = zip(self.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    self.load_state_dict(state_dict, strict=True)
    
class FedPid(FedAvg):
    def __init__(self, fraction_fit: float = 1.0, *args, **kwargs):
        self.num_rounds = kwargs.pop('num_rounds', 60)  # Default to 30 if not provided
        super().__init__(*args, **kwargs)
        self.fraction_fit = fraction_fit # starts with 1, in principle
        self.last_global_weights = None
        self.last_local_loss = {}
        self.client_prob_by_cid = {}
        self.last_distributed_loss = None
        self.num_participating_clients = None
        self.setpoint = 0.05
        self.current_distributed_acc = 0  # Initialize accuracy tracking
        
        # PID gains
        self.Kp = 1.0  # Proportional gain
        self.Ki = 0.1  # Integral gain
        self.Kd = 0.05  # Derivative gain
        
        logging.info(f'Initializing PID with Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}')
        
        # Initialize the PID controller
        self.pid = PID(self.Kp, self.Ki, self.Kd, setpoint=self.setpoint)
        self.csv_file_path = os.path.join(HydraConfig.get().runtime.output_dir, 'pid_metrics.csv')

        
        # Write CSV header if the file doesn't exist
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Round',  'Fraction Fit', 'Distributed Accuracy', 'Process Variable', 'Setpoint'])
        
       
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
        total_clients = len(clients)
        
        # Prepare the fit instructions for the selected clients
        
        if server_round == 1:
            fit_ins = FitIns(parameters, config)
            client_instructions = [(client, fit_ins) for client in clients]
            # Calculate the minimum fraction fit required to select at least 2 clients
            min_fraction_fit = 2 / total_clients
            self.pid.output_limits = (min_fraction_fit, 1)  # Set limits for the number of clients
            
        else:
            logging.info(f'Aggregated evaluation accuracy for server_round {server_round} is: {self.current_distributed_acc}')
            logging.info(f'fraction fit for round {server_round} is  {self.fraction_fit}')
            
            process_variable = (1 - self.current_distributed_acc) * (self.fraction_fit)
        
            logging.info(f'Value of process_variable for round {server_round} is: {process_variable}')
            
            logging.info(f'calling update pid control')
            self.fraction_fit = self.pid(process_variable)
            
            logging.info(f'fraction_fit calculated from pid for {server_round} is: {self.fraction_fit}')
            
            if(self.fraction_fit > 1):
                logging.info(f'Limiting fraction_fit 1')
                self.fraction_fit = 1
                
            # For subsequent rounds, select clients based on their local loss
            sorted_clients = sorted(
                clients,
                key=lambda client: self.last_local_loss.get(client.cid, float('-inf')),  # Sort by loss value or use a large negative number if not found
                reverse=True  # Highest loss first
            )
            
            # Calculate the number of fit clients for the current round based on fraction fit
            self.num_participating_clients = int(self.fraction_fit*total_clients)
                    
            logging.info(f'Calculated Number of clients for round {server_round} is {self.num_participating_clients}')
            
            if self.num_participating_clients < 2:
                logging.info(f'Updating num_participating to the minimum value of 2')
                self.num_participating_clients = 2
            
            # Select the top clients based on the number of clients to participate
            selected_clients = sorted_clients[:self.num_participating_clients]
            
            # Create FitIns for selected clients
            fit_ins = FitIns(parameters, config)
            client_instructions = [(client, fit_ins) for client in selected_clients]
            
            # Save metrics to the CSV file
            with open(self.csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([server_round, self.fraction_fit, self.current_distributed_acc, process_variable, self.setpoint])
                
        return client_instructions
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        parameters_aggregated, metrics = super().aggregate_fit(server_round,results,failures)
        
        # current_loss = metrics['local_loss']
        # current_distributed_acc = metrics['local_acc']
       
        logging.info('Started updating last_local_loss dict')
        
        # update dict with last local loss info
        for (client, fit_res) in results:
            cid = client.cid
            self.last_local_loss[cid] = fit_res.metrics['local_loss']
            
        logging.info('Finished updating self.last_local_loss dict')

        return parameters_aggregated, {}
    
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
    
        # Call the base class method to perform the usual aggregation
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        # Check if accuracy was calculated and present in metrics_aggregated
        if "accuracy" in metrics_aggregated:
            self.current_distributed_acc = metrics_aggregated["accuracy"]
            logging.info(f'Aggregated evaluation accuracy for server_round {server_round} is: {self.current_distributed_acc}')
        else:
            logging.warning(f"No accuracy metric found for server_round {server_round}. Setting current_distributed_acc to 0.")
            self.current_distributed_acc = 0  # Fallback in case accuracy isn't available
        
        # Return the aggregated evaluation results
        return loss_aggregated, metrics_aggregated
    
    
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
        
    
