#!/bin/env bash
module purge

# Declare the Python script name as a variable
python_script_name="test.py"

# echo "Outside of apptainer, host python version:"
# python3 --version
# apptainer exec ~/RydbergGPT/images/pytorch.sif echo "This is from inside a container. Check python version:"
# apptainer exec ~/RydbergGPT/images/pytorch.sif python --version
# apptainer exec ~/RydbergGPT/images/pytorch.sif python ~/RydbergGPT/examples/${python_script_name}

apptainer exec ~/RydbergGPT/images/pytorch.sif python ~/RydbergGPT/${python_script_name} --config_name=config_small