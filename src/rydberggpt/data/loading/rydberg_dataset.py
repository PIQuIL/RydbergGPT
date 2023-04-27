import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from rydberggpt.utils import to_one_hot


def get_rydberg_dataloader(
    batch_size: int = 32,
    test_size: float = 0.2,
    dataset_name: str = "dataset",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Generate DataLoaders for the train and validation sets of a Rydberg atom dataset.

    Args:
        batch_size (int, optional): The batch size for the DataLoader. Default is 32.
        test_size (float, optional): The proportion of the dataset to be used as validation set. Default is 0.2.
        dataset_name (str, optional): The filename of the dataset without the file extension. Default is "dataset".
        num_workers (int, optional): The number of workers to use for data loading. Default is 0.

    Returns:
        tuple: A tuple containing the train and validation DataLoaders.
            train_loader (DataLoader): The DataLoader for the train dataset.
            val_loader (DataLoader): The DataLoader for the validation dataset.
    """
    df = pd.read_hdf(f"data/{dataset_name}.h5", key="data")
    # Split the DataFrame into train and validation DataFrames
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Create the RydbergDataset for train and validation sets
    train_dataset = RydbergDataset(train_df)
    val_dataset = RydbergDataset(test_df)

    # Create a DataLoader for each set with the custom Dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


class RydbergDataset(Dataset):
    """
    A custom Dataset class for handling Rydberg atom data.

    Attributes:
        dataframe (pd.DataFrame): The input dataframe containing Rydberg atom data.
    """

    def __init__(self, dataframe):
        """
        Initialize the RydbergDataset with the given dataframe.

        Args:
            dataframe (pd.DataFrame): The input dataframe containing Rydberg atom data.
        """
        self.dataframe = dataframe

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """
        Get the data sample at the specified index.

        Args:
            index (int): The index of the data sample.

        Returns:
            tuple: A tuple containing the conditions, shifted measurements, and measurements tensors.
                cond (torch.Tensor): A tensor representing the input conditions. Shape: [1, 4].
                shifted_measurements (torch.Tensor): A tensor representing the shifted measurements. Shape: [N, 2].
                measurements (torch.Tensor): A tensor representing the original measurements. Shape: [N, 2].
        """
        row = self.dataframe.iloc[index]
        cond = torch.tensor(
            [row["delta"], row["omega"], row["Lx"], row["Ly"]], dtype=torch.float32
        )
        cond = cond.unsqueeze(0)  # [batch_size, 1, 4] 1 is the sequ length dim
        measurements = torch.tensor(row["measurement"], dtype=torch.int64)
        measurements = to_one_hot(measurements, 2)  # because Rydberg states are 0 or 1

        _, dim = measurements.shape
        shifted_measurements = torch.cat(
            (torch.zeros(1, dim), measurements[:-1]), dim=0
        )
        return cond, shifted_measurements, measurements
