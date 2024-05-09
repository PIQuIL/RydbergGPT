# Usage

## Configuration
The`config.yaml` is used to define the hyperparameters for :
- Model architecture
- Training settings
- Data loading
- Others

## Training
To train RydbergGPT locally, execute the `train.py` with :
```bash
python train.py --config_name=config_small.yaml
```
For the cluster:
```bash
.scripts/train.sh
```
