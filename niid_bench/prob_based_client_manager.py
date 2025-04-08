from flwr.server.client_manager import SimpleClientManager
import numpy as np

class ProbabilityBasedClientManager(SimpleClientManager):
    def __init__(self, initial_probabilities=None):
        super().__init__()
        self.probabilities = initial_probabilities
    
    def normalize_probabilities(self):
        if self.probabilities is not None:
            self.probabilities = self.probabilities / np.sum(self.probabilities)
    
    def assign_probabilities(self, similarity_scores):
        self.probabilities = similarity_scores
        
        self.normalize_probabilities()
    
    def sample_clients(self, num_clients):
        # Sample clients based on assigned probabilities
        clients = list(self.all().values())
        
        client_ids = [client.cid for client in clients]
        
        selected_client_ids = np.random.choice(client_ids, size=num_clients, p=self.probabilities)
        
        selected_clients = [self.get(client_id) for client_id in selected_client_ids]
        
        return selected_clients