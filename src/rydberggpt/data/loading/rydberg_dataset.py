import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def get_rydberg_dataloader(batch_size=32, test_size=0.2):
    df = pd.read_hdf("data/dataset.h5", key="data")
    # Split the DataFrame into train and validation DataFrames
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Create the RydbergDataset for train and validation sets
    train_dataset = RydbergDataset(train_df)
    val_dataset = RydbergDataset(test_df)

    # Create a DataLoader for each set with the custom Dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class RydbergDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        delta_omega_lx_ly = torch.tensor(
            [row["delta"], row["omega"], row["Lx"], row["Ly"]], dtype=torch.float32
        )
        measurement = torch.tensor(row["measurement"], dtype=torch.int64)
        return delta_omega_lx_ly, measurement
