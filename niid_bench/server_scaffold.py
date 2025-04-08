"""Server class for SCAFFOLD."""

import concurrent.futures
from logging import DEBUG, INFO
from typing import OrderedDict

import torch
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import (
    Callable,
    Dict,
    GetParametersIns,
    List,
    NDArrays,
    Optional,
    Tuple,
    Union,
)
from flwr.server import Server
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import numpy as np

from niid_bench.models import test

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]

import logging


class ScaffoldServer(Server):
    """Implement server for SCAFFOLD."""

    def __init__(
        self,
        strategy: Strategy,
        model: DictConfig,
        client_manager: Optional[ClientManager] = None,
    ):
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.model_params = instantiate(model)
        self.server_cv: List[torch.Tensor] = []

    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from one of the available clients."""
        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout)
        log(INFO, "Received initial parameters from one random client")
        self.server_cv = [
            torch.from_numpy(t)
            for t in parameters_to_ndarrays(get_parameters_res.parameters)
        ]
        return get_parameters_res.parameters

    # pylint: disable=too-many-locals
    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strateg
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=update_parameters_with_cv(self.parameters, self.server_cv),
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[Optional[Parameters], Dict[str, Scalar]] = (
            self.strategy.aggregate_fit(server_round, results, failures)
        )

        aggregated_result_arrays_combined = []
        if aggregated_result[0] is not None:
            aggregated_result_arrays_combined = parameters_to_ndarrays(
                aggregated_result[0]
            )
        aggregated_parameters = aggregated_result_arrays_combined[
            : len(aggregated_result_arrays_combined) // 2
        ]
        aggregated_cv_update = aggregated_result_arrays_combined[
            len(aggregated_result_arrays_combined) // 2 :
        ]

        # convert server cv into ndarrays
        server_cv_np = [cv.numpy() for cv in self.server_cv]
        # update server cv
        total_clients = len(self._client_manager.all())
        cv_multiplier = len(results) / total_clients
        self.server_cv = [
            torch.from_numpy(cv + cv_multiplier * aggregated_cv_update[i])
            for i, cv in enumerate(server_cv_np)
        ]

        # update parameters x = x + 1* aggregated_update
        curr_params = parameters_to_ndarrays(self.parameters)
        updated_params = [
            x + aggregated_parameters[i] for i, x in enumerate(curr_params)
        ]
        parameters_updated = ndarrays_to_parameters(updated_params)

        # metrics
        metrics_aggregated = aggregated_result[1]
        return parameters_updated, metrics_aggregated, (results, failures)


def update_parameters_with_cv(
    parameters: Parameters, s_cv: List[torch.Tensor]
) -> Parameters:
    """Extend the list of parameters with the server control variate."""
    # extend the list of parameters arrays with the cv arrays
    cv_np = [cv.numpy() for cv in s_cv]
    parameters_np = parameters_to_ndarrays(parameters)
    parameters_np.extend(cv_np)
    return ndarrays_to_parameters(parameters_np)


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout)
    return client, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    model: DictConfig,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generate the function for centralized evaluation.

    Parameters
    ----------
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
               Optional[Tuple[float, Dict[str, Scalar]]] ]
    The centralized evaluation function.
    """

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        """Use the entire Emnist test set for evaluation."""
        net = instantiate(model)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        loss, accuracy = test(net, testloader, device=device)
        return loss, {"accuracy": accuracy}

    return evaluate



def gen_ensemble_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    model: DictConfig,
    cluster_models_list: List[Parameters]
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generate the evaluation function for an ensemble of models."""
    
    def evaluate(
        server_round: int, cluster_models_list: List[Parameters], config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate each cluster model and select the most confident predictions."""

        logging.info(f"Starting evaluation for round {server_round}")
        num_clusters = len(cluster_models_list)
        total_loss, total_accuracy = 0.0, 0.0
        
        # Collect logits from each model for each test sample
        all_logits = []

        logging.info("Collecting logits from each model in the ensemble")
        
        for idx, cluster_params in enumerate(cluster_models_list):
            net = instantiate(model)
            params_dict = zip(net.state_dict().keys(), parameters_to_ndarrays(cluster_params))
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            net.to(device)
            
            # Ensure the model is in evaluation mode (disables Dropout, BatchNorm, etc.)
            net.eval()
            
            logging.info(f"Evaluating model {idx} for round {server_round}")
            
            # Perform evaluation for this model and store logits
            logits = collect_logits(net, testloader, device)
            logging.info(f"Model {idx} logits collected for {len(logits)} samples")
            all_logits.append(logits)
        
        logging.info("Logits collected from all models, proceeding to select highest logit predictions")

        # Select the model with the highest logit for each test sample
        ensemble_predictions = select_highest_logit_predictions(all_logits)
        logging.info(f"Predictions selected from models with highest logits")

        # Calculate accuracy based on the most confident predictions
        final_accuracy = calculate_accuracy(ensemble_predictions, testloader)
        logging.info(f"Final accuracy for round {server_round}: {final_accuracy}")

        return None, {"accuracy": final_accuracy}

    return evaluate

def collect_logits(net, testloader, device):
    """Evaluate the model and return the logits for each test sample."""
    net.eval()
    all_logits = []

    logging.info("Collecting logits from model during inference")

    with torch.no_grad():
        for data, _ in testloader:
            data = data.to(device)
            output = net(data)  # Get raw logits (no softmax here)
            logging.debug(f"Logits shape: {output.shape}")  # Debug log for shape of logits
            all_logits.append(output.cpu().numpy())  # Store logits for each sample

    logging.info(f"Logits collected for {len(all_logits)} batches")
    return np.concatenate(all_logits)

def select_highest_logit_predictions(all_logits):
    """Select predictions based on the highest logit value from the ensemble."""
    logging.info("Selecting predictions based on highest logit value")

    # Stack logits for all models, shape: (num_models, num_samples, num_classes)
    stacked_logits = np.stack(all_logits, axis=0)
    logging.debug(f"Stacked logits shape: {stacked_logits.shape}")  # Debug for stacked shape

    # For each sample, find the model that gives the highest logit for its predicted class
    # We need to take the maximum value along the model dimension (axis=0)
    
    # Find the max logits along the model axis (axis=0), shape: (num_samples, num_classes)
    max_logits = np.max(stacked_logits, axis=0)

    # Now, select the class with the highest logit for each sample, shape: (num_samples,)
    ensemble_predictions = np.argmax(max_logits, axis=1)

    logging.info(f"Predictions generated based on highest logit")
    return ensemble_predictions

def calculate_accuracy(ensemble_predictions, testloader):
    """Calculate the accuracy of the ensemble predictions."""
    logging.info("Calculating accuracy for ensemble predictions")

    true_labels = []
    for _, target in testloader:
        true_labels.extend(target.numpy())

    true_labels = np.array(true_labels)
    correct = (ensemble_predictions == true_labels).sum()
    accuracy = correct / len(true_labels)

    logging.info(f"Accuracy calculated: {accuracy}")
    return accuracy