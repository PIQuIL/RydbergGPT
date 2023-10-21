import logging

from rydberggpt.data.loading.rydberg_dataset import get_rydberg_dataloader
from rydberggpt.utils import time_and_log

# setup logging
logging.basicConfig(level=logging.INFO)


@time_and_log
def main():
    base_dir = "src/rydberggpt/tests/dataset_test/"

    dataloader, _ = get_rydberg_dataloader(
        batch_size=128, data_path=base_dir, buffer_size=2
    )

    counter = 0
    for batch in dataloader:
        counter += batch.m_onehot.shape[0]
        # print(batch.m_onehot.shape)

    print(counter)


if __name__ == "__main__":
    main()
