import logging
from typing import Any, Tuple

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def set_example_input_array(train_loader: DataLoader) -> Tuple[Any, Any]:
    """
    Get an example input array from the train loader.

    Args:
        train_loader (DataLoader): The DataLoader instance for the training data.

    Returns:
        (Tuple[Any, Any]): A tuple containing m_onehot and graph from the example batch.
    """
    logging.info("Setting example input array...")
    example_batch = next(iter(train_loader))
    return example_batch.m_onehot, example_batch.graph
