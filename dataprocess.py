import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Example DataFrame
data_dict = {
    "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
    "feature2": [5.0, 4.0, 3.0, 2.0, 1.0],
    "label": [0, 1, 0, 1, 0],
}
df = pd.DataFrame(data_dict)

class PandasDataset(Dataset):
    def __init__(self, dataframe, feature_columns, label_column):
        """
        Args:
            dataframe (pd.DataFrame): The input DataFrame.
            feature_columns (list): List of column names used as features.
            label_column (str): Column name for the labels.
        """
        self.dataframe = dataframe
        self.features = dataframe[feature_columns].values
        self.labels = dataframe[label_column].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get features and label for a given index
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # For classification tasks
        return features, label

# Define feature and label columns
feature_columns = ["feature1", "feature2"]
label_column = "label"

# Create Dataset
dataset = PandasDataset(df, feature_columns, label_column)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Iterate through DataLoader
for batch_idx, (features, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}")
    print(f"Features: \n{features}")
    print(f"Labels: \n{labels}")

