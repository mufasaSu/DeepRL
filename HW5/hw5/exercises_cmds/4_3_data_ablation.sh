#!/bin/bash

export PYTORCH_ENABLE_MPS_FALLBACK=1

python ./cs285/scripts/run_hw5_offline.py \
    -cfg experiments/offline/pointmass_medium_cql.yaml \
    --dataset_dir datasets_ablation/1k/ &

# Wait for 2 minutes
sleep 60

python ./cs285/scripts/run_hw5_offline.py \
    -cfg experiments/offline/pointmass_medium_cql.yaml \
    --dataset_dir datasets_ablation/5k/ &

wait

python ./cs285/scripts/run_hw5_offline.py \
    -cfg experiments/offline/pointmass_medium_cql.yaml \
    --dataset_dir datasets_ablation/10k/ &

# Wait for 2 minutes
sleep 60

python ./cs285/scripts/run_hw5_offline.py \
    -cfg experiments/offline/pointmass_medium_cql.yaml \
    --dataset_dir datasets_ablation/20k/
    