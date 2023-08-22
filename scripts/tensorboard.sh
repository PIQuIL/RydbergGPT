module purge

apptainer exec ~/RydbergGPT/images/pytorch.sif tensorboard --logdir="logs/" --port=6007