import pytest
import torch
from tqdm import tqdm

from rydberggpt.data.loading.rydberg_dataset_chunked_random import (
    get_chunked_random_dataloader,
)


# Define a pytest fixture that sets up the dataloader for the test.
# This fixture will be created once (due to scope="module") and passed to any test that requests it.
@pytest.fixture(scope="module")
def chunked_random_dataloader():
    # Define the directory where test datasets are located
    base_dir = "src/rydberggpt/tests/dataset_test/"

    # Set batch size for the dataloader
    batch_size = 100

    # Create and return the dataloader
    dataloader, _ = get_chunked_random_dataloader(
        batch_size=batch_size, data_path=base_dir, chunks_in_memory=3
    )
    return dataloader


def test_corrupted_data_samples(chunked_random_dataloader):
    """
    Test to ensure there are no corrupted data samples in the dataset.

    This test checks the following:
    - The batch size of m_onehot is as expected.
    - The dimension of m_onehot is correct.
    - m_onehot and m_shifted_onehot have the same shape.
    - No NaN values are present in m_onehot.
    """

    # Loop through the dataset multiple times to ensure consistency across different chunks
    for i in range(2):
        for batch in tqdm(chunked_random_dataloader):
            # Ensure the batch size of m_onehot is 100
            assert batch.m_onehot.shape[0] == 100, "Batch size of m_onehot is not 100."

            # Ensure the third dimension of m_onehot is 2
            assert batch.m_onehot.shape[2] == 2, "Dimension of m_onehot is not 2."

            # Ensure m_onehot and m_shifted_onehot have the same shape
            assert (
                batch.m_onehot.shape == batch.m_shifted_onehot.shape
            ), "Shapes of m_onehot and m_shifted_onehot are not the same."

            # Ensure the required attributes are present in the batch
            assert hasattr(batch, "graph")
            assert hasattr(batch, "m_onehot")
            assert hasattr(batch, "m_shifted_onehot")

            # Check for NaN values in m_onehot and raise an assertion if any are found
            assert not torch.isnan(
                batch.m_onehot
            ).any(), "NaN values found in m_onehot."

            # Clean up the current batch to free memory
            del batch


if __name__ == "__main__":
    pytest.main([__file__])
