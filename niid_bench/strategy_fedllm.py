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
import requests

API_KEY = '13qd759fs4'

from niid_bench.utils import calculate_similarity,find_clusters, experiment_with_dbscan_and_print


initial_message = [
    {
        "role": "system",
        "content": (
            "You are an expert federated learning assistant. Consider all the knowledge about client selection for federated learning existent on the scientific literature"
            "Your task is to help with client selection at each training round. "
            "At each round, you will receive performance data from all clients. "
            "Your goal is to minimize the number of updates (client participation) "
            "while improving the global model’s centralized accuracy. "
            "Based on the data, return the number of clients to select, and the client IDs of selected clients."
            
        )
    },
    
]

output_format_instructions = (
    "Select the appropriate number of clients to participate in the next round. "
    "Return only a valid JSON object in the following format containing the number of clients selected, the client ids of the selected clients, "
    "and a brief explanation of why those clients were selected:\n"
    "{\n"
    "  \"num_clients\": X,\n"
    "  \"selected_clients\": [\"id\", \"id2\"],\n"
    "  \"explanation\": \"Your explanation here.\"\n"
    "}\n"
    "You are free to select any clients in any round, regardless of their participation history. "
    "Avoid selecting the same clients every round unless strictly necessary. Consider exploring other clients when possible to help improve the global model more efficiently over time. "
    "Only output the JSON object in this exact format, and do not include any extra text outside the JSON."
)

def set_parameters(self, parameters):
    """Set the local model parameters using given ones."""
    params_dict = zip(self.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    self.load_state_dict(state_dict, strict=True)
    
def invoke_jamba(messages, api_key):
    url = "https://tfrq6se0fg.execute-api.us-east-1.amazonaws.com/invoke_model"

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    payload = {
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.8
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(response.json())
    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": "Request failed",
            "status_code": response.status_code,
            "response": response.text
        }
    
    
class FedLLM(FedAvg):
    def __init__(self, fraction_fit: float = 1.0, *args, **kwargs):
        self.num_rounds = kwargs.pop('num_rounds', 30)  # Default to 30 if not provided
        super().__init__(*args, **kwargs)
        self.fraction_fit = fraction_fit
        self.last_global_weights = None
        self.conversation_messages = initial_message
        self.communication_cost = 0
        self.global_accuracy = 'N/A'
        
    def evaluate(self, server_round: int, parameters: Parameters
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    
        """Evaluate model parameters using the appropriate evaluation function."""
    
        # Bypass evaluation in round 0
        if server_round == 0:
            logging.info("Skipping evaluation for round 0.")
            return None
        
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
    
        if eval_res is None:
            logging.info(f"Eval is none")
            return None

        
        loss, metrics = eval_res
        print(f'metrics  ealuate: {metrics}')
        self.global_accuracy = metrics.get("accuracy", 0.0)
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
            # sample clients from each cluster as per fraction fit
            sampled_clients = self.sample_clients_via_llm(server_round,clients)
            
            client_instructions = [] 
            for client in sampled_clients:
                cid = int(client.cid)
                config['cid'] = cid
                fit_ins = FitIns(parameters, config)
                client_instructions.append((client, fit_ins))
                
                
        return client_instructions
    
    def sample_clients_via_llm(self, server_round: int, clients: List[ClientProxy]) -> List[ClientProxy]:
        """
        Selects clients for training using LLM-guided selection.

        This function sends the conversation history to the LLM (stored in self.conversation_messages),
        parses the JSON response, and filters the clients accordingly.
        """

        logging.info(f"[Round {server_round}] Querying LLM for client selection...")

        try:
            # Call LLM with conversation history
            response = invoke_jamba(self.conversation_messages, API_KEY)

            if "reply" not in response:
                raise ValueError("LLM response missing 'reply' field")

            # Parse JSON string returned by LLM
            llm_reply = json.loads(response["reply"])
            selected_ids = [str(int(cid.replace("client_", ""))) for cid in llm_reply.get("selected_clients", [])]
            num_selected = llm_reply.get("num_clients", len(selected_ids))

            # Log response
            explanation = llm_reply.get("explanation", "No explanation provided.")
            logging.info(f"[LLM] Selected {num_selected} clients: {selected_ids} in Round {server_round}, explanation is:\n {explanation}")
            
            

            # Add assistant response to conversation
            self.conversation_messages.append({
                "role": "assistant",
                "content": json.dumps(llm_reply)
            })

            # Filter the clients by ID
            selected_clients = [client for client in clients if client.cid in selected_ids]

            if not selected_clients:
                logging.warning("[LLM] No clients matched the selection. Falling back to all clients.")
                return clients

            return selected_clients

        except Exception as e:
            logging.error(f"[LLM] Failed to sample clients via LLM: {e}")
            logging.warning("[LLM] Falling back to all clients.")
            return clients
        
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[List[Parameters]], Dict[str, Scalar]]:

        # Initialize string for LLM message
        round_client_data = ""

        for client, fit_res in results:
            cid = int(client.cid)
            metrics = fit_res.metrics

            similarity = metrics.get("similarity", "N/A")
            loss = metrics.get("loss", "N/A")

            # Log and collect for prompt
            logging.info(f"[Round {server_round}] Client {cid} - similarity with global model: {similarity}, local loss: {loss}")
            round_client_data += f"- client_{cid:02d}: similarity={similarity}, loss={loss}\n"
            self.communication_cost += 1 # Increment communication cost for each client

        # Aggregate as usual
        global_model, metrics = super().aggregate_fit(server_round, results, failures)
        
        # Updating partictipation summary
        participation_summary = self.update_participation(results)

        print(f"Global accuracy: {self.global_accuracy}\n")
        print(f"Current Communication Cost: {self.communication_cost}\n")
        print(f"Client Participation Summary {participation_summary}\n")
        


        # Append formatted prompt to LLM conversation
        self.conversation_messages.append({
            "role": "user",
            "content": (
                f"Round {server_round} results:\n"
                f"Global accuracy: {self.global_accuracy}\n"
                f"Current Communication Cost: {self.communication_cost}\n"
                f"Client Participation Summary {participation_summary}\n"
                f"{round_client_data}\n"
                f"{output_format_instructions}"
            )
        })

        logging.info("Aggregation complete and LLM message updated.")
        return global_model, metrics
    
    def update_participation(self, results: List[Tuple[ClientProxy, FitRes]]) -> str:
        """Track client participation and return a formatted summary string."""
        if not hasattr(self, "client_participation"):
            self.client_participation = defaultdict(int)

        for client, _ in results:
            cid = int(client.cid)
            self.client_participation[cid] += 1

        summary = "Client participation count so far:\n"
        for cid, count in sorted(self.client_participation.items()):
            summary += f"- client_{cid:02d}: {count} rounds\n"

        return summary

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
        
    
