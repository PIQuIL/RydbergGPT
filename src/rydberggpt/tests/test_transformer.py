import pytest
from rydberggpt.data.dataclasses import Batch
from rydberggpt.data.rydberg_dataset import get_rydberg_dataloader
from rydberggpt.models.rydberg_encoder_decoder import get_rydberg_graph_encoder_decoder
from rydberggpt.training.trainer import RydbergGPTTrainer
from rydberggpt.utils import create_config_from_yaml, load_yaml_file


@pytest.fixture(scope="module")
def config_dataloader():
    return {"data_path": "src/rydberggpt/tests/dataset_test/", "batch_size": 10}


def test_transformer_with_dataloader(config_dataloader):
    yaml_dict = load_yaml_file("config/", "config_small")
    config = create_config_from_yaml(yaml_dict)
    dataloader = get_rydberg_dataloader(**config_dataloader)

    model = get_rydberg_graph_encoder_decoder(config)

    rydberg_gpt_trainer = RydbergGPTTrainer(model, config)

    for i, batch in enumerate(dataloader):
        assert isinstance(batch, Batch), "Batch is not an instance of the Batch class"

        # Perform a forward pass through the model using the trainer
        output = rydberg_gpt_trainer.training_step(batch, i)

        assert output >= 0, "Loss is not non-negative"

        # Exit loop after n batches for quick testing
        if i >= 10:
            break


pytest.main([__file__])
