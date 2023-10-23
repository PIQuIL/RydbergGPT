import logging

from tqdm import tqdm

from rydberggpt.data.dataclasses import Batch
from rydberggpt.data.loading.rydberg_dataset_2 import get_rydberg_dataloader_2
from rydberggpt.utils import time_and_log

# setup logging
logging.basicConfig(level=logging.INFO)


@time_and_log
def main():
    batch_size = 128
    buffer_size = 2
    num_workers = 1
    base_dir = "src/rydberggpt/tests/dataset_test/"

    dataloader, _ = get_rydberg_dataloader_2(
        batch_size=batch_size,
        data_path=base_dir,
        buffer_size=buffer_size,
        num_workers=num_workers,
    )

    counter = 0
    for batch in tqdm(dataloader):
        if counter == 0:
            assert (
                batch.m_onehot.shape[0] == batch_size
            ), f"Batch size of m_onehot is not {batch_size}."
            assert batch.m_onehot.shape[2] == 2, "Dimension of m_onehot is not 2."
            assert (
                batch.m_onehot.shape == batch.m_shifted_onehot.shape
            ), "Shapes of m_onehot and m_shifted_onehot are not the same."
            assert isinstance(batch, Batch)
            assert hasattr(batch, "graph")
            assert hasattr(batch, "m_onehot")
            assert hasattr(batch, "m_shifted_onehot")

        counter += batch.m_onehot.shape[0]
    print(counter)


if __name__ == "__main__":
    main()
