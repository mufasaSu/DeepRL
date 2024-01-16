#!/bin/bash

python cs285/scripts/run_hw5_explore.py \
    -cfg experiments/exploration/pointmass_easy_rnd.yaml \
    --dataset_dir datasets/ &

python cs285/scripts/run_hw5_explore.py \
    -cfg experiments/exploration/pointmass_medium_rnd.yaml \
    --dataset_dir datasets/ &

wait

python cs285/scripts/run_hw5_explore.py \
    -cfg experiments/exploration/pointmass_hard_rnd.yaml \
    --dataset_dir datasets/ &


wait 

export PYTORCH_ENABLE_MPS_FALLBACK=1

# IQL Medium
python ./cs285/scripts/run_hw5_offline.py \
    -cfg experiments/offline/pointmass_medium_iql.yaml \
    --dataset_dir datasets/ &

# AWAC Easy
python ./cs285/scripts/run_hw5_offline.py \
    -cfg experiments/offline/pointmass_easy_awac.yaml \
    --dataset_dir datasets/

wait

# AWAC Medium
python ./cs285/scripts/run_hw5_offline.py \
    -cfg experiments/offline/pointmass_medium_awac.yaml \
    --dataset_dir datasets/