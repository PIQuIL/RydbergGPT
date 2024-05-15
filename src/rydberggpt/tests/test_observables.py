import os
from dataclasses import dataclass

import torch
from rydberggpt.models.rydberg_encoder_decoder import get_rydberg_graph_encoder_decoder
from rydberggpt.models.utils import generate_prompt
from rydberggpt.observables.rydberg_energy import (
    get_rydberg_energy,
    get_staggered_magnetization,
    get_x_magnetization,
)
from rydberggpt.utils import create_config_from_yaml, load_yaml_file
from rydberggpt.utils_ckpt import get_model_from_ckpt
from torch_geometric.data import Batch

device = "cuda" if torch.cuda.is_available() else "cpu"

base_path = os.path.abspath(".")
log_path = os.path.join(base_path, "models/M_1/")

yaml_dict = load_yaml_file(log_path, "hparams.yaml")
config: dataclass = create_config_from_yaml(yaml_dict)

model = get_model_from_ckpt(
    log_path, model=get_rydberg_graph_encoder_decoder(config), ckpt="best"
)
model.to(device=device)
model.eval()  # don't forget to set to eval mode


def test_observables():
    L = 5
    delta = 1.0
    omega = 1.0
    beta = 64.0
    Rb = 1.15
    num_samples = 5

    pyg_graph = generate_prompt(
        model_config=config,
        n_rows=L,
        n_cols=L,
        delta=delta,
        omega=omega,
        beta=beta,
        Rb=Rb,
    )

    # duplicate the prompt for num_samples
    cond = [pyg_graph for _ in range(num_samples)]
    cond = Batch.from_data_list(cond)

    samples = model.get_samples(
        batch_size=len(cond), cond=cond, num_atoms=L**2, fmt_onehot=False
    )

    energy = get_rydberg_energy(model, samples, cond=pyg_graph, device=device)
    print(energy.mean() / L**2)

    staggered_magnetization = get_staggered_magnetization(samples, L, L, device=device)
    print(staggered_magnetization.mean() / L**2)

    x_magnetization = get_x_magnetization(model, samples, cond=pyg_graph, device=device)
    print(x_magnetization.mean() / L**2)
    assert True


test_observables()
