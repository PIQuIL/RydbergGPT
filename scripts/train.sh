#!/bin/env bash

#SBATCH -A NAISS2023-5-353  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 03-00:00:00
#SBATCH --gpus-per-node=A100:3

module purge

# Declare the Python script name as a variable
python_script_name="train.py"
# python_script_name="examples/3_train_encoder_decoder.py"

apptainer exec ~/RydbergGPT/images/pytorch.sif python ~/RydbergGPT/${python_script_name} --config_name=config_small