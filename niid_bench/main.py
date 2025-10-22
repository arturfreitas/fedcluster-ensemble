"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import os
import pickle

import flwr as fl
import hydra
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

from niid_bench.dataset import load_datasets
from niid_bench.server_fednova import FedNovaServer
from niid_bench.server_scaffold import ScaffoldServer, gen_evaluate_fn, gen_ensemble_evaluate_fn
from niid_bench.strategy import FedNovaStrategy, ScaffoldStrategy
from niid_bench.strategy import FedAvgGenetic
import datetime
import pandas as pd
from niid_bench.utils import weighted_average

from niid_bench.strategy_fedcosinecluster import FedCosineCluster
import logging
from typing import List, Tuple
from hydra import compose, initialize

def main_wrapper(fraction_fit_lst: List[float]) -> Tuple[float, float]:
    # context initialization
    with initialize(version_base=None, config_path="conf", job_name="test_app"):
        cfg = compose(config_name="fedavg_genetic")
        print(OmegaConf.to_yaml(cfg))
    # Import cfg DictConfig variable from the yaml file
    #cfg = hydra.compose(config_name="fedavg_genetic")

    # Add the fraction_fit_lst directly to the configuration
    cfg.fraction_fit_lst = fraction_fit_lst

    # Call main passing cfg and retrieving centralized accuracy and communication cost
    fitness_tuple = main(cfg)  # (accuracy, cost)

    return fitness_tuple
    
    

@hydra.main(config_path="conf", config_name="fedavg_base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    if "mnist" in cfg.dataset_name:
        cfg.model.input_dim = 256
        # pylint: disable=protected-access
        cfg.model._target_ = "niid_bench.models.CNNMnist"
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    trainloaders, valloaders, testloader = load_datasets(
        config=cfg.dataset,
        num_clients=cfg.num_clients,
        val_ratio=cfg.dataset.val_split,
    )

    # 3. Define your clients
    client_fn = None
    # pylint: disable=protected-access
    if cfg.client_fn._target_ == "niid_bench.client_scaffold.gen_client_fn":
        save_path = HydraConfig.get().runtime.output_dir
        client_cv_dir = os.path.join(save_path, "client_cvs")
        print("Local cvs for scaffold clients are saved to: ", client_cv_dir)
        client_fn = call(
            cfg.client_fn,
            trainloaders,
            valloaders,
            model=cfg.model,
            client_cv_dir=client_cv_dir,
        )
    else:
        client_fn = call(
            cfg.client_fn,
            trainloaders,
            valloaders,
            model=cfg.model,
        )

    device = cfg.server_device
    
    # Step 1: Instantiate the strategy from the configuration
    strategy = instantiate(cfg.strategy)
    
    # Step 2: Check if the strategy is FedCosineCluster and set the appropriate evaluate function
    if isinstance(strategy, FedCosineCluster):
        logging.info("Using ensemble evaluation function")
        evaluate_fn = gen_ensemble_evaluate_fn(testloader, device, cfg.model, strategy.cluster_label_to_model_map)
    else:
        logging.info("Using traditional evaluate function")
        evaluate_fn = gen_evaluate_fn(testloader, device, cfg.model)
        
    if isinstance(strategy, FedAvgGenetic):
        strategy.fraction_fit_lst = cfg.fraction_fit_lst

    # Step 3: Set the evaluate_fn attribute on the strategy
    strategy.evaluate_fn = evaluate_fn

    # Step 4: Optionally, set other attributes on the strategy
    strategy.fit_metrics_aggregation_fn = weighted_average
    
    strategy.evaluate_metrics_aggregation_fn = weighted_average

    # 5. Define your server
    server = Server(strategy=strategy, client_manager=SimpleClientManager())
    if isinstance(strategy, FedNovaStrategy):
        server = FedNovaServer(strategy=strategy, client_manager=SimpleClientManager())
    elif isinstance(strategy, ScaffoldStrategy):
        server = ScaffoldServer(
            strategy=strategy, model=cfg.model, client_manager=SimpleClientManager()
        )

    # 6. Start Simulation
    history = fl.simulation.start_simulation(
        #server=server,
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        strategy=strategy,
    )
    
    
    if isinstance(strategy, FedAvgGenetic):
        cost = sum(value for _, value in history.metrics_distributed_fit['fraction_fit'])
        centralized_accuracy = history.metrics_centralized['accuracy'][-1][1]
        
        
        return centralized_accuracy,cost
    
    print(type(history))

    # Convert the history object to a pandas DataFrame
    #history_df = convert_to_df(history)

    print("history metrics distributed:")
    print(history.metrics_distributed)
    
    print("history metrics centralized:")
    print(history.metrics_centralized)
    
    distributed_history_dict = {}
    centralized_history_dict = {}
    for metric, round_value_tuple_list in history.metrics_distributed.items():
        distributed_history_dict["aggregated_test_" + metric] = [
            val for _, val in round_value_tuple_list
        ]
    for metric, round_value_tuple_list in history.metrics_distributed_fit.items():  # type: ignore
        distributed_history_dict["aggregateed_" + metric] = [
            val for _, val in round_value_tuple_list
        ]
    # distributed_history_dict["distributed_test_loss"] = [
    #     val for _, val in history.losses_distributed
    # ]
    
    for metric, round_value_tuple_list in history.metrics_centralized.items():  # type: ignore
        centralized_history_dict[metric] = [
            val for _, val in round_value_tuple_list
        ]
    
        # Define the save path using HydraConfig
    save_path = HydraConfig.get().runtime.output_dir

    # Define the output file name with the timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    distributed_history_filename = f"distributed_metrics.csv"
    centralized_history_filename = f"centralized_metrics.csv"


    print(distributed_history_dict)
    distributed_history_df = pd.DataFrame.from_dict(distributed_history_dict)
    distributed_history_df.to_csv(f"{save_path}/{distributed_history_filename}", index=False)
    
    centralized_history_df = pd.DataFrame.from_dict(centralized_history_dict)
    centralized_history_df.to_csv(f"{save_path}/{centralized_history_filename}", index=False)
    
    

    # Print the save path (optional)
    print(f"Results saved to: {save_path}")

    # 7. Save your results
    # with open(os.path.join(save_path, "history.pkl"), "wb") as f_ptr:
    #     pickle.dump(history, f_ptr)
        
    # Save the DataFrame as a CSV file


if __name__ == "__main__":
    main()
