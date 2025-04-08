import pandas as pd

class ExperimentResult:
    def __init__(self, **kwargs):
        # Dynamically set attributes from CSV data
        self.__dict__.update(kwargs)
        self.centralized_metrics = None
        self.client_participation = None
        self.distributed_metrics = None
        self.load_data()

    def load_data(self):
        # Load specific experiment data from CSV files in the output folder
        try:
            self.centralized_metrics = pd.read_csv(f"{self.output_folder}/centralized_metrics.csv")
            #self.distributed_metrics = pd.read_csv(f"{self.output_folder}/distributed_metrics.csv")
            #self.client_participation = pd.read_csv(f"{self.output_folder}/client_participation_by_round.csv")
            
        except  Exception as e:
            print(f'{e}')



def load_experiments_from_csv(csv_file_path):
    """Load experiments from a CSV file and return a list of ExperimentResult objects."""
    experiments = []
    df = pd.read_csv(csv_file_path)
    for _, row in df.iterrows():
        if row.get('valid', 'No') == 'Yes':  # Only include experiments marked as 'Yes' in the 'valid' column
            exp = ExperimentResult(**row.to_dict())
            experiments.append(exp)
    return experiments

if __name__ == "__main__":
    csv_file_path = "experiments_fl.csv"
    experiments = load_experiments_from_csv(csv_file_path)
    
    print(experiments)