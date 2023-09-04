import pytest

from rydberggpt.data.dataclasses import Batch
from rydberggpt.data.loading import (
    get_chunked_dataloader,
    get_chunked_random_dataloader,
    get_rydberg_dataloader,
    get_streaming_dataloader,
)
from rydberggpt.models.rydberg_encoder_decoder import get_rydberg_graph_encoder_decoder
from rydberggpt.training.trainer import RydbergGPTTrainer
from rydberggpt.utils import create_config_from_yaml, load_yaml_file


# Define a fixture for common model parameters (if needed)
@pytest.fixture(scope="module")
def config():
    return {
        "config_path": "configs/",
    }


# Define a fixture for common parameters
@pytest.fixture(scope="module")
def config_dataloader():
    return {"data_path": "src/rydberggpt/tests/dataset_test/", "batch_size": 10}


def test_rydberg_gpt_with_dataloader(config_dataloader):
    yaml_dict = load_yaml_file("config/", "config_small")
    config = create_config_from_yaml(yaml_dict)
    dataloader, _ = get_chunked_random_dataloader(**config_dataloader)

    # Create Model
    model = get_rydberg_graph_encoder_decoder(config)

    # Initialize the trainer
    rydberg_gpt_trainer = RydbergGPTTrainer(model, config)

    for i, batch in enumerate(dataloader):
        assert isinstance(batch, Batch), "Batch is not an instance of the Batch class"

        # Perform a forward pass through the model using the trainer
        output = rydberg_gpt_trainer.training_step(batch, i)

        # Check output shapes, values, or any other property you are interested in
        assert output >= 0, "Loss is not non-negative"

        # Exit loop after one batch for quick testing
        if i >= 1:
            break


if __name__ == "__main__":
    pytest.main([__file__])

    # config_dataloader = {
    #     "data_path": "src/rydberggpt/tests/dataset_test/",
    #     "batch_size": 10,
    # }

    # test_rydberg_gpt_with_dataloader(config_dataloader)
