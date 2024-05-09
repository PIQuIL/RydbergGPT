import logging

import torch
from rydberggpt.data.dataclasses import Batch
from rydberggpt.data.rydberg_dataset import get_rydberg_dataloader
from rydberggpt.utils import shift_inputs, time_and_log
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


@time_and_log
def test_dataloader():
    batch_size = 128
    buffer_size = 2
    num_workers = 0
    base_dir = "src/rydberggpt/tests/dataset_test/"

    dataloader = get_rydberg_dataloader(
        batch_size=batch_size,
        data_path=base_dir,
        buffer_size=buffer_size,
        num_workers=num_workers,
    )

    counter = 0
    for batch in tqdm(dataloader):
        assert (
            batch.m_onehot.shape[0] == batch_size
        ), f"Batch size of m_onehot is not {batch_size}."
        assert batch.m_onehot.shape[2] == 2, "Dimension of m_onehot is not 2."

        m_shifted_onehot = shift_inputs(batch.m_onehot)
        assert (
            batch.m_onehot.shape == m_shifted_onehot.shape
        ), "Shapes of m_onehot and m_shifted_onehot are not the same."

        assert (batch.m_onehot[:, :-1, :] == m_shifted_onehot[:, 1:, :]).all()
        assert isinstance(batch, Batch)
        assert hasattr(batch, "graph")
        assert hasattr(batch, "m_onehot")
        assert (
            batch.m_onehot.shape[0] == batch_size
        ), "Batch size of m_onehot is not 100."
        assert batch.m_onehot.shape[2] == 2, "Dimension of m_onehot is not 2."
        assert not torch.isnan(batch.m_onehot).any(), "NaN values found in m_onehot."

        counter += batch.m_onehot.shape[0]
    print(counter)


test_dataloader()
