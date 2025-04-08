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
import random


from niid_bench.utils import calculate_similarity,find_clusters, experiment_with_dbscan_and_print

def set_parameters(self, parameters):
    """Set the local model parameters using given ones."""
    params_dict = zip(self.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    self.load_state_dict(state_dict, strict=True)
    
class FedCosineCluster(FedAvg):
    def __init__(self, fraction_fit: float = 1.0, *args, **kwargs):
        self.num_rounds = kwargs.pop('num_rounds', 30)  # Default to 30 if not provided
        super().__init__(*args, **kwargs)
        self.fraction_fit = fraction_fit
        self.last_global_weights = None
        self.cluster_label_to_model_map = {} # dict mapping cluster label to cluster model (Parameters objects returned from fedavg aggregate_fit)
        self.cid_cluster_map = {}
        
    def evaluate(self, server_round: int, parameters: Parameters
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    
        """Evaluate model parameters using the appropriate evaluation function."""
    
        # Bypass evaluation in round 0
        if server_round == 0:
            logging.info("Skipping evaluation for round 0.")
            return None

        # If no evaluation function is provided, return None
        if self.evaluate_fn is None:
            return None

        # Perform ensemble evaluation for clusters
        if self.cluster_label_to_model_map:
            logging.info("Performing ensemble evaluation for clusters.")
        
            # Use the list of cluster models for ensemble evaluation
            cluster_models_list = list(self.cluster_label_to_model_map.values())
            eval_res = self.evaluate_fn(server_round, cluster_models_list, {})
        else:
            # If no cluster models are available, fallback to traditional evaluation
            logging.info("No cluster models found, performing traditional evaluation.")
            parameters_ndarrays = parameters_to_ndarrays(parameters)
            eval_res = super().evaluate_fn(server_round, parameters_ndarrays, {})
    
        if eval_res is None:
            return None

        loss, metrics = eval_res
        return loss, metrics
       
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
        # Calculate the number of clients for the current round
       
        logging.info(f'Number of clients for round {server_round} is {num_clients}')

        # Prepare the fit instructions for the selected clients
        
        if server_round == 1:
        
            fit_ins = FitIns(parameters, config)
            client_instructions = [(client, fit_ins) for client in clients]
        else:
            
            logging.info(f'Sampling clients from clusters, fraction fit is {self.fraction_fit}')
            
            # sample clients from each cluster as per fraction fit
            sampled_clients = self.sample_cluster(server_round,clients)
            
            client_instructions = []
            # assign model to client according with 
            logging.info('printing length of cluster_label_to_model_map')
            logging.info(len(self.cluster_label_to_model_map))
            
            for client in sampled_clients:
                cid = int(client.cid)
                config['cid'] = cid
                client_cluster_label = self.cid_cluster_map[cid]
                parameters_client = self.cluster_label_to_model_map[client_cluster_label]
                fit_ins = FitIns(parameters_client, config)
                client_instructions.append((client, fit_ins))
                
            
      
                
        return client_instructions
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[List[Parameters]], Dict[str, Scalar]]:
        
        # cluster assignment takes place only on first round, since the training data is stationary
        if server_round == 1:
        
            last_layers_with_cid = [] # List to hold tuples of (cid, last_layer_weights)

            for client, fit_res in results:
                # Request the current model from the client
                client_params = fit_res.parameters
                client_weights = parameters_to_ndarrays(client_params)
                last_layer_weights = client_weights[-1]  # Extract last layer weights
                cid = int(client.cid)

                # Append tuple (cid, last_layer_weights) to the list
                last_layers_with_cid.append((cid, last_layer_weights)) 

            logging.info(f'calling find clusters for weights')

            # to do experiment with different clustering parameters
            #experiment_with_dbscan_and_print(last_layers_with_cid)
           
            self.cid_cluster_map, silhouette_avg = find_clusters(last_layers_with_cid)
        
            logging.info("Clustering results:")
            logging.info(self.cid_cluster_map)
            logging.info(f'silhouette score: {silhouette_avg}')
            #pdb.set_trace()
        
        # following logic will happen every round
        
        cluster_results_map = defaultdict(list) # dict {int} = Union[Tuple[ClientProxy, FitRes]]

        logging.info('populating cluster_results_map with results of each cluster member')
        
        for (client, fit_res) in results:
            client_cid_int = int(client.cid)
            cluster_label = self.cid_cluster_map[client_cid_int] 
            cluster_results_map[cluster_label].append((client, fit_res))
            
        logging.info('Finished populating cluster_results_map!')
        
        # Aggregating results of each cluster and generating cluster models
        
        logging.info('Peforming aggregation for each cluster')
        # Perform aggregation for each cluster separately (FedAvg)
        for cluster_label, cluster_results in cluster_results_map.items():
            logging.info(f'Aggregating model for cluster {cluster_label} with {len(cluster_results)} results')
            cluster_model, cluster_metrics = super().aggregate_fit(server_round,cluster_results,failures)
            logging.info(f'metrics for cluster  {cluster_label}: {cluster_metrics}')
            self.cluster_label_to_model_map[cluster_label] = cluster_model
            
       
        # create a list of models for ensemble use
        cluster_models_list = list(self.cluster_label_to_model_map.values())
        
        logging.info('Returning results')
        return cluster_models_list[0], {}
     
    
    def aggregate_clusters_to_global(self, cluster_label_to_model_map: List[NDArrays]) -> NDArrays:
        """Combine cluster-specific models into a final global model."""
        # Simple average of cluster models (can be replaced with more complex methods if needed)
        num_clusters = len(cluster_label_to_model_map)
        final_global_model = [np.zeros_like(layer) for layer in parameters_to_ndarrays(cluster_label_to_model_map[0])]

        for label, cluster_model in cluster_label_to_model_map.items():
            cluster_model_ndarrays = parameters_to_ndarrays(cluster_model)
            for i, layer in enumerate(cluster_model_ndarrays):
                final_global_model[i] += layer / num_clusters

        final_global_model = ndarrays_to_parameters(final_global_model)
        
        return final_global_model
    


    def sample_cluster(self, server_round, clients):
        """
        Samples clients from clusters based on the fraction of clients to fit.

        Args:
            server_round (int): The current server round in the federated learning process.
            clients (list): A list of all available clients in the current round.
            cyclic_mode (bool, optional): If True, will use cyclic sampling instead of random sampling. Defaults to False.

        Returns:
            list: A list of sampled clients based on their cluster association.
        """
        # Dictionary that maps each cluster to a list of its clients
        cluster_clients_lst_map = defaultdict(list)
    
        # Lists for storing sampled client IDs and clients
        sampled_cids = []
        sampled_clients = []
    
        logging.info('Creating cluster label to list of client IDs')
    
        # Convert `cid_cluster_map` (client ID to cluster label) into `cluster_clients_lst_map` (cluster label to list of client IDs)
        for cid, cluster_label in self.cid_cluster_map.items():
            cluster_clients_lst_map[cluster_label].append(cid)
            
        
        logging.info('Iterating through clusters and sampling clients from each cluster ')
    
        # Iterate through clusters and sample clients from each cluster
        for cluster, client_ids in cluster_clients_lst_map.items():
            
                
            if self.fraction_fit == 'cyclic':
                index = int((server_round - 2)%len(client_ids))
                logging.info(f'Cyclic mode: selected index {index} for cluster {cluster}')
                sampled_cluster = [client_ids[index]] # wrap single client on a list
            else:
                
                logging.info(f'Random sampling with fraction_fit {self.fraction_fit}')
                
                # Calculate the number of clients to sample
                num_clients_to_sample = max(1, int(self.fraction_fit * len(client_ids)))
                
                # Sample clients randomly from the cluster
                sampled_cluster = random.sample(client_ids, num_clients_to_sample)
            
            sampled_cids.extend(sampled_cluster)
        
        logging.info(f'Sampled client_ids: {sampled_cids}')

        logging.info('Filtering clients based on sampled clients IDS')
    
        # Filter clients based on sampled client IDs
        sampled_clients = [client for client in clients if int(client.cid) in sampled_cids]

        # returning client objects
        return sampled_clients

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
        
    
