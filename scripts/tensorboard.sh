#!/bin/env bash


module purge
FREE_PORT=`comm -23 <(seq "8888" "8988" | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf -n 1`
apptainer exec ~/RydbergGPT/images/pytorch.sif tensorboard --logdir="logs/" --port=${FREE_PORT}