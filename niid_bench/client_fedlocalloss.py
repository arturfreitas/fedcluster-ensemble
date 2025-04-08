"""Defines the client class and support functions for FedAvg."""

from typing import Callable, Dict, List, OrderedDict

import flwr as fl
import torch
from flwr.common import Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from niid_bench.models import test, train_fedavg, train_fedavg_with_loss
from collections import Counter
from typing import Dict
from niid_bench.utils import calculate_similarity
import sys
import logging
# pylint: disable=too-many-instance-attributes
class FlowerClientFedLocalLoss(fl.client.NumPyClient):
    """Flower client implementing FedAvg."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Implement distributed fit function for a given client for FedAvg."""
        
        self.set_parameters(parameters)
        
        train_fedavg(
            self.net,
            self.trainloader,
            self.device,
            self.num_epochs,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
        )
        
        final_p_np = self.get_parameters({})
        
        server_round = config["server_round"]
        
        similarity = calculate_similarity(parameters[-1], final_p_np[-1])
        data_transferred = sys.getsizeof(final_p_np)
        
        # Calculate the class distribution in the training dataset
        class_distribution = self.get_class_distributions()
        
        loss, acc = test(self.net, self.trainloader, self.device)
        
        return_dict = {
            "similarity": similarity,
            "dataset_size": len(self.trainloader),
            "data_transferred": data_transferred,
            "data_distribution": class_distribution,
            'local_loss': loss
        }
    
        return final_p_np, len(self.trainloader.dataset), return_dict

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        #logging.info(f"Starting evaluate for {config['cid']}")
        self.set_parameters(parameters)
        loss, acc = test(self.net, self.valloader, self.device)
        
        # Calculate the class distribution in the training dataset
        # class_distribution = self.get_class_distributions_val()
        
        # print(f"printing client distribution for client:")
        # print(class_distribution)
        #print(f"Evalulate Accuracy on round {config['server_round']}  is {acc}, length of data is {len(self.valloader.dataset)}")
        
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}
    
    def get_class_distributions(self):
        # Define the number of classes
        num_classes = 10
    
        # Initialize a list of zeros for all class counts
        class_counts = [0] * num_classes
    
        # Calculate the class distribution in the training dataset
        for _, label in self.trainloader.dataset:
            class_counts[label] += 1
    
        return class_counts
    
    def get_class_distributions_val(self):
        # Define the number of classes
        num_classes = 10
    
        # Initialize a list of zeros for all class counts
        class_counts = [0] * num_classes
    
        # Calculate the class distribution in the training dataset
        for _, label in self.valloader.dataset:
            class_counts[label] += 1
    
        return class_counts


# pylint: disable=too-many-arguments
def gen_client_fn(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    num_epochs: int,
    learning_rate: float,
    model: DictConfig,
    momentum: float = 0.9,
    weight_decay: float = 1e-5,
) -> Callable[[str], FlowerClientFedLocalLoss]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the FedAvg flower clients.

    Parameters
    ----------
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    valloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.
    momentum : float
        The momentum for SGD optimizer of clients
    weight_decay : float
        The weight decay for SGD optimizer of clients

    Returns
    -------
    Callable[[str], FlowerClientFedAvg]
        The client function that creates the FedAvg flower clients
    """

    def client_fn(cid: str) -> FlowerClientFedLocalLoss:
        """Create a Flower client representing a single organization."""
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClientFedLocalLoss(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            momentum,
            weight_decay,
        )

    return client_fn
