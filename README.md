# FedClusterEnsemble: Clustered Ensemble Federated Learning under Non-IID Data

This repository implements FedClusterEnsemble, a clustered ensemble framework for Federated Learning (FL) designed to improve training stability and communication efficiency under label-skewed Non-IID client data.

Clients are grouped into clusters based on data similarity, and a separate model is trained per cluster. During training, adaptive client participation reduces redundant updates within clusters. At inference time, predictions are produced using a confidence-based ensemble that selects or combines cluster-specific models.

Key Features

Similarity-based client clustering

One model per cluster to mitigate Non-IID effects

Adaptive participation policies to reduce communication cost

Confidence-based ensemble inference

Analysis of convergence stability, communication efficiency, and client fairness

Experimental Setup

Datasets: MNIST, Fashion-MNIST, CIFAR-10, SVHN

Task: Image classification

Non-IID Settings: Dirichlet label skew, fixed labels per client, label-group partitions

Baselines: FedAvg, SCAFFOLD, FedProx, FedNova

Results Overview

Across multiple datasets and Non-IID configurations, FedClusterEnsemble demonstrates:

More stable convergence than FedAvg

Lower communication overhead via adaptive participation

Competitive accuracy under strong label skew

Acknowledgement

This implementation is built on top of the NIID-Bench experimental framework
(Qinbin Li et al., Federated Learning on Non-IID Data Silos: An Experimental Study, arXiv:2102.02079).

``` 
