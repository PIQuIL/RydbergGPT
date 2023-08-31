import pytest

from rydberggpt.data.dataclasses import Batch  # Assuming Batch is imported from here
from rydberggpt.data.loading.rydberg_dataset import get_rydberg_dataloader
from rydberggpt.data.loading.rydberg_dataset_chunked import get_chunked_dataloader
from rydberggpt.data.loading.rydberg_dataset_streaming import get_streaming_dataloader


class TestRydbergDatasets:
    def __init__(self):
        self.data_path = "src/rydberggpt/tests/dataset_test/"
        self.batch_size = 12

    def test_dataloader_common(self, dataloader):
        for batch in dataloader:
            print(batch)
            assert (
                batch.m_onehot.shape[0] == self.batch_size
            ), f"Batch size of m_onehot is not {self.batch_size}."
            assert batch.m_onehot.shape[2] == 2, "Dimension of m_onehot is not 2."
            assert (
                batch.m_onehot.shape == batch.m_shifted_onehot.shape
            ), "Shapes of m_onehot and m_shifted_onehot are not the same."
            assert isinstance(batch, Batch)
            assert hasattr(batch, "graph")
            assert hasattr(batch, "m_onehot")
            assert hasattr(batch, "m_shifted_onehot")
            break

    def test_get_rydberg_dataloader(self):
        dataloader, _ = get_rydberg_dataloader(
            data_path=self.data_path, batch_size=self.batch_size
        )
        self.test_dataloader_common(dataloader)

    def test_get_chunked_dataloader(self):
        dataloader, _ = get_chunked_dataloader(
            data_path=self.data_path, batch_size=self.batch_size
        )
        self.test_dataloader_common(dataloader)

    def test_get_streaming_dataloader(self):
        dataloader, _ = get_streaming_dataloader(
            data_path=self.data_path, batch_size=self.batch_size
        )
        self.test_dataloader_common(dataloader)


if __name__ == "__main__":
    pytest.main([__file__])
