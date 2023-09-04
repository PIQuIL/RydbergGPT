import pytest

from rydberggpt.data.dataclasses import Batch  # Assuming Batch is imported from here
from rydberggpt.data.loading import (
    get_chunked_dataloader,
    get_chunked_random_dataloader,
    get_rydberg_dataloader,
    get_streaming_dataloader,
)


# Define a fixture for common parameters
@pytest.fixture(scope="module")
def common_parameters():
    return {"data_path": "src/rydberggpt/tests/dataset_test/", "batch_size": 10}


# Define your fixtures for each dataloader type, and use the common_parameters fixture as an argument
@pytest.fixture(scope="module")
def rydberg_dataloader(common_parameters):
    dataloader, _ = get_rydberg_dataloader(**common_parameters)
    return dataloader


@pytest.fixture(scope="module")
def chunked_dataloader(common_parameters):
    dataloader, _ = get_chunked_dataloader(**common_parameters)
    return dataloader


@pytest.fixture(scope="module")
def chunked_random_dataloader(common_parameters):
    dataloader, _ = get_chunked_random_dataloader(**common_parameters)
    return dataloader


@pytest.fixture(scope="module")
def streaming_dataloader(common_parameters):
    dataloader, _ = get_streaming_dataloader(**common_parameters)
    return dataloader


# Parameterize the common test to run for each dataloader
@pytest.mark.parametrize(
    "dataloader",
    [
        "rydberg_dataloader",
        "chunked_dataloader",
        "chunked_random_dataloader",
        "streaming_dataloader",
    ],
)
def test_dataloader_common(request, dataloader):
    dataloader_instance = request.getfixturevalue(dataloader)

    for batch in dataloader_instance:
        assert batch.m_onehot.shape[0] == 10, "Batch size of m_onehot is not 10."
        assert batch.m_onehot.shape[2] == 2, "Dimension of m_onehot is not 2."
        assert (
            batch.m_onehot.shape == batch.m_shifted_onehot.shape
        ), "Shapes of m_onehot and m_shifted_onehot are not the same."
        assert isinstance(batch, Batch)
        assert hasattr(batch, "graph")
        assert hasattr(batch, "m_onehot")
        assert hasattr(batch, "m_shifted_onehot")
        break


if __name__ == "__main__":
    pytest.main([__file__])
