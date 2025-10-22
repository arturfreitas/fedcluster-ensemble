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
    """
    Load experiments from a CSV file and return a list of ExperimentResult objects.

    ⚠️ On macOS and Unix systems, file paths should use '/' instead of '\\'. This function normalizes paths automatically.
    """
    experiments = []
    df = pd.read_csv(csv_file_path)

    for _, row in df.iterrows():
        if row.get('valid', 'No') == 'Yes':  # Only include experiments marked as 'Yes' in the 'valid' column
            row_dict = row.to_dict()

            # Normalize Windows-style paths to Unix-style (for macOS compatibility)
            if 'output_folder' in row_dict and isinstance(row_dict['output_folder'], str):
                row_dict['output_folder'] = row_dict['output_folder'].replace("\\", "/")

            exp = ExperimentResult(**row_dict)
            experiments.append(exp)

    return experiments

if __name__ == "__main__":
    csv_file_path = "experiments_fl.csv"
    experiments = load_experiments_from_csv(csv_file_path)
    
    print(experiments)