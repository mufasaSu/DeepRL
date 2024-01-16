#!/bin/bash

export PYTORCH_ENABLE_MPS_FALLBACK=1

# AWAC
python ./cs285/scripts/run_hw5_offline.py \
    -cfg experiments/offline/pointmass_medium_iql.yaml \
    --dataset_dir datasets/ &


# IQL
python ./cs285/scripts/run_hw5_offline.py \
    -cfg experiments/offline/pointmass_easy_awac.yaml \
    --dataset_dir datasets/