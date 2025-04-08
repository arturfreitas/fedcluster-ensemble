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
from niid_bench.utils import calculate_gini_coefficient
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
from scipy.special import softmax
from niid_bench.utils import export_dict_to_csv
from hydra.core.hydra_config import HydraConfig
import pandas as pd

def set_parameters(self, parameters):
    """Set the local model parameters using given ones."""
    params_dict = zip(self.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    self.load_state_dict(state_dict, strict=True)
    
    


class FedNovaStrategy(FedAvg):
    """Custom FedAvg strategy with fednova based configuration and aggregation."""

    def aggregate_fit_custom(
        self,
        server_round: int,
        server_params: NDArrays,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        total_samples = sum([fit_res.num_examples for _, fit_res in results])
        c_fact = sum(
            [
                float(fit_res.metrics["a_i"]) * fit_res.num_examples / total_samples
                for _, fit_res in results
            ]
        )
        new_weights_results = [
            (result[0], c_fact * (fit_res.num_examples / total_samples))
            for result, (_, fit_res) in zip(weights_results, results)
        ]

        # Aggregate grad updates, t_eff*(sum_i(p_i*\eta*d_i))
        grad_updates_aggregated = aggregate_fednova(new_weights_results)
        # Final parameters = server_params - grad_updates_aggregated
        aggregated = [
            server_param - grad_update
            for server_param, grad_update in zip(server_params, grad_updates_aggregated)
        ]

        parameters_aggregated = ndarrays_to_parameters(aggregated)
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

class FedCosine(FedAvg):
    def __init__(self, fraction_fit: float = 1.0, *args, **kwargs):
        self.num_rounds = kwargs.pop('num_rounds', 100)  # Default to 100 if not provided
        super().__init__(*args, **kwargs)
        self.fraction_fit = fraction_fit
        self.last_global_weights = None
        
        # Extract num_rounds from kwargs
        
        
        self.client_participation_data = {}
        
        logging.info(f"Starting training with num_rounds equal{self.num_rounds}")
        
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
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
        
        # Prepare the fit instructions for the selected clients
        fit_ins = FitIns(parameters, {})
        
        # Convert parameters to weights (assuming model is using weights)
        global_weights = parameters_to_ndarrays(parameters)
        self.last_global_weights = global_weights
        
        # Get all clients
        all_clients = list(client_manager.all().values())
        
        # For the first round, include all clients
        client_instructions = [(client, fit_ins) for client in all_clients ]
        
        if server_round == 1:
            # Update the client participation data for the first round
            all_clients_cids = [client.cid for client in all_clients]
            self.client_participation_data[server_round] = ", ".join(map(str, all_clients_cids))
            
        
        return client_instructions
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        client_similarities = []
        
        if server_round > 1:
            for client, fit_res in results:
                # Request the current model from the client
                client_params = fit_res.parameters
                client_weights = parameters_to_ndarrays(client_params)

                # Calculate similarity (e.g., cosine similarity) between the last layers
                similarity = self.calculate_similarity(self.last_global_weights[-1], client_weights[-1])
                client_similarities.append((client.cid, similarity))

            # Sort clients by similarity
            client_similarities.sort(key=lambda x: x[1], reverse=True)

            # Step 2: Calculate the number of clients to select (top 70%)
            num_clients_to_select = int(0.7 * len(client_similarities))

            # Step 3: Select the top 70% of clients
            selected_clients = client_similarities[:num_clients_to_select]
            
            # Step 5: Create a list of selected client IDs
            selected_client_ids = [client_id for client_id, _ in selected_clients]
            selected_client_ids.sort()
            
            logging.info(f'[fedosine] selected clients for round {server_round} are: {selected_client_ids}')
            
            # Update the client participation data for this round
            self.client_participation_data[server_round] = ", ".join(map(str, selected_client_ids))
            
            # Filter the results to include only the selected clients
            results_filtered = [(client, fit_res) for client, fit_res in results if client.cid in selected_client_ids]
        else:
            # In the first round, use all clients
            results_filtered = results

        # Perform the aggregation (e.g., FedAvg) to update the global model
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results_filtered, failures)
        
        if server_round > 1:
            logging.info(f'Client similarities for round number {server_round}')

            # Print client similarities in the desired format
            for cid, similarity in client_similarities:
               logging.info(f"{cid} : {similarity}")
                
                
        if server_round == self.num_rounds:
            save_path = HydraConfig.get().runtime.output_dir
            df = pd.DataFrame(list(self.client_participation_data.items()), columns=['round', 'participating_clients'])
            df.to_csv(f'{save_path}/client_participation_by_round.csv', index=False)

        return parameters_aggregated, metrics_aggregated
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average and calculate Gini score."""
        
        # Call the super method to perform the default aggregation
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Calculate Gini score based on client accuracies
        accuracies = [
            evaluate_res.metrics['accuracy'] for _, evaluate_res in results if 'accuracy' in evaluate_res.metrics
        ]
        gini_score = calculate_gini_coefficient(accuracies) if accuracies else None

        # Include the Gini score in the aggregated metrics
        if gini_score is not None:
            metrics_aggregated['gini_score'] = gini_score

        return loss_aggregated, metrics_aggregated
    

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
    
    

        
class FedCosineDivergence(FedCosine):
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        client_similarities = []
        
        if server_round > 1:
            for client, fit_res in results:
                # Request the current model from the client
                client_params = fit_res.parameters
                client_weights = parameters_to_ndarrays(client_params)

                # Calculate similarity (e.g., cosine similarity) between the last layers
                #shape of client_weights[-1] is (10,)
                similarity = self.calculate_similarity(self.last_global_weights[-1], client_weights[-1])
                client_similarities.append((client.cid, similarity))

            # Sort clients by similarity
            client_similarities.sort(key=lambda x: x[1], reverse=False)

            # Step 2: Calculate the number of clients to select (top 70%)
            num_clients_to_select = int(0.7 * len(client_similarities))

            # Step 3: Select the top 70% of clients
            selected_clients = client_similarities[:num_clients_to_select]
            
            # Step 5: Create a list of selected client IDs
            selected_client_ids = [client_id for client_id, _ in selected_clients]
            selected_client_ids.sort()
            
            logging.info(f'selected clients for round {server_round} are: {selected_client_ids}')
            
            # Update the client participation data for this round
            self.client_participation_data[server_round] = ", ".join(map(str, selected_client_ids))
            
            # Filter the results to include only the selected clients
            results_filtered = [(client, fit_res) for client, fit_res in results if client.cid in selected_client_ids]
            
            
        else:
            # In the first round, use all clients
            results_filtered = results

        # Perform the aggregation (e.g., FedAvg) to update the global model
        parameters_aggregated, metrics_aggregated = super(FedCosine, self).aggregate_fit(server_round, results_filtered, failures)
        
        if server_round > 1:
            logging.info(f'Client similarities for round number {server_round}')

            # Print client similarities in the desired format
            for cid, similarity in client_similarities:
               logging.info(f"{cid} : {similarity}")
                
                
        if server_round == self.num_rounds:
            save_path = HydraConfig.get().runtime.output_dir
            df = pd.DataFrame(list(self.client_participation_data.items()), columns=['round', 'participating_clients'])
            df.to_csv(f'{save_path}/client_participation_by_round.csv', index=False)

        return parameters_aggregated, metrics_aggregated
    

def aggregate_fednova(results: List[Tuple[NDArrays, float]]) -> NDArrays:
    """Implement custom aggregate function for FedNova."""
    # Create a list of weights, each multiplied by the weight_factor
    weighted_weights = [
        [layer * factor for layer in weights] for weights, factor in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


class ScaffoldStrategy(FedAvg):
    """Implement custom strategy for SCAFFOLD based on FedAvg class."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        combined_parameters_all_updates = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        len_combined_parameter = len(combined_parameters_all_updates[0])
        num_examples_all_updates = [fit_res.num_examples for _, fit_res in results]
        # Zip parameters and num_examples
        weights_results = [
            (update[: len_combined_parameter // 2], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        # Aggregate parameters
        parameters_aggregated = aggregate(weights_results)

        # Zip client_cv_updates and num_examples
        client_cv_updates_and_num_examples = [
            (update[len_combined_parameter // 2 :], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        aggregated_cv_update = aggregate(client_cv_updates_and_num_examples)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return (
            ndarrays_to_parameters(parameters_aggregated + aggregated_cv_update),
            metrics_aggregated,
        )
        

class CustomFedAvg(FedAvg):
    def __init__(self, fraction_fit: float = 1.0, *args, **kwargs):
        self.num_rounds = kwargs.pop('num_rounds', 100)  # Default to 100 if not provided
        super().__init__(*args, **kwargs)
        self.fraction_fit = fraction_fit

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average and calculate Gini score."""
        
        # Call the super method to perform the default aggregation
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Calculate Gini score based on client accuracies
        accuracies = [
            evaluate_res.metrics['accuracy'] for _, evaluate_res in results if 'accuracy' in evaluate_res.metrics
        ]
        gini_score = calculate_gini_coefficient(accuracies) if accuracies else None

        # Include the Gini score in the aggregated metrics
        if gini_score is not None:
            metrics_aggregated['gini_score'] = gini_score

        return loss_aggregated, metrics_aggregated
    
    
class FedAvgGenetic(FedAvg):
    def __init__(self, fraction_fit: float = 1.0, *args, **kwargs):
        self.num_rounds = kwargs.pop('num_rounds', 30)  # Default to 100 if not provided
        self.fraction_fit_lst = kwargs.pop('fraction_fit_lst',[])
        super().__init__(*args, **kwargs)
        
        self.fraction_fit = fraction_fit
        
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    )    ->     List[Tuple[ClientProxy, FitIns]]:
    
     config = {}
     config['server_round'] = server_round

     # Get all clients
     clients = list(client_manager.all().values())
     num_clients = len(clients)

    

     current_fraction_fit = self.fraction_fit_lst[server_round-1]

     # Calculate the number of clients to sample based on the fraction_fit
     num_clients_sampled = int(current_fraction_fit * num_clients) # It might be zero

     logging.info(f'Number of clients for round {server_round} is {num_clients_sampled}')

     # Prepare the fit instructions for the selected clients

     logging.info(f'Sampling clients randomly according to fraction fit, fraction fit is {self.fraction_fit}')

     # Sample clients randomly according to the fraction fit
     sampled_clients = random.sample(clients, num_clients_sampled)

     client_instructions = []


     for client in sampled_clients:
         cid = int(client.cid)
         config['cid'] = cid
         fit_ins = FitIns(parameters, config)
         client_instructions.append((client, fit_ins))

     return client_instructions

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average and calculate Gini score."""
        
        # Call the super method to perform the default aggregation
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Calculate Gini score based on client accuracies
        accuracies = [
            evaluate_res.metrics['accuracy'] for _, evaluate_res in results if 'accuracy' in evaluate_res.metrics
        ]
        gini_score = calculate_gini_coefficient(accuracies) if accuracies else None

        # Include the Gini score in the aggregated metrics
        if gini_score is not None:
            metrics_aggregated['gini_score'] = gini_score

        return loss_aggregated, metrics_aggregated
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitIns]],
        failures: List[Union[Tuple[ClientProxy, FitIns], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate fit results and add the fraction_fit metric."""
        
        # Call the super method to perform the default aggregation
        loss_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # Calculate the fraction_fit for this round
        current_fraction_fit = self.fraction_fit_lst[server_round - 1]  # Adjusting for zero-based index

        # Add the fraction_fit metric to the aggregated metrics
        metrics_aggregated['fraction_fit'] = current_fraction_fit

        return loss_aggregated, metrics_aggregated
    
    

