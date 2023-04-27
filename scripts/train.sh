#!/bin/env bash

#SBATCH -A SNIC2022-5-398  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 00-00:15:00
#SBATCH --gpus-per-node=T4:3

module purge

# Declare the Python script name as a variable
python_script_name="4_pytorch_lightning.py"
# python_script_name="3_train_encoder_decoder.py"

apptainer exec ~/RydbergGPT/images/pytorch.sif python ~/RydbergGPT/examples/${python_script_name} --config_name=gpt2