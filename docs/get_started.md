# Get Started

## Installation

Clone the repository using the following command:
```bash
git clone https://github.com/PIQuIL/RydbergGPT
```
Install with pip :
```bash
cd RydbergGPT
pip install .
```

## Usage
### Tutorials
You find tutorials for using RydbergGPT in the [`examples/`](https://github.com/PIQuIL/RydbergGPT/tree/main/examples) folder.

### Configuration
The`config.yaml` is used to define the hyperparameters for:
- Model architecture
- Training settings
- Data loading
- Others

### Training
To train RydbergGPT locally, execute the `train.py` with: 
```bash
python train.py --config_name=config_small.yaml
```