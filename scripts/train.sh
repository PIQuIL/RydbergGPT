#!/bin/env bash

#SBATCH -A NAISS2023-5-353  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 01-00:00:00 # DD-HH:MM:SS
#SBATCH --gpus-per-node=A100fat:1
#SBATCH -J A100 #job_name 
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail          # send email when job ends
#SBATCH --mail-user=davidfi@chalmers.se

module purge

# Declare the Python script name as a variable
python_script_name="train.py"
config_name="config_small"
# python_script_name="examples/3_train_encoder_decoder.py"

apptainer exec ~/RydbergGPT/images/pytorch.sif python ~/RydbergGPT/${python_script_name} --config_name=${config_name} --dataset_path=dataset