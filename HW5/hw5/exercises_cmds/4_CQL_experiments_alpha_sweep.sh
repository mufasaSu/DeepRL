#!/bin/bash

# Old experiments (turns out an alpha of > 1 performs bad):

# python ./cs285/scripts/run_hw5_offline.py \
#     -cfg experiments/offline/cql_alpha_experiments/pointmass_medium_cql_2.yaml \
#     --dataset_dir datasets/ &

# python ./cs285/scripts/run_hw5_offline.py \
#     -cfg experiments/offline/cql_alpha_experiments/pointmass_medium_cql_4.yaml \
#     --dataset_dir datasets/ &

# wait

# python ./cs285/scripts/run_hw5_offline.py \
#     -cfg experiments/offline/cql_alpha_experiments/pointmass_medium_cql_6.yaml \
#     --dataset_dir datasets/ &

# python ./cs285/scripts/run_hw5_offline.py \
#     -cfg experiments/offline/cql_alpha_experiments/pointmass_medium_cql_8.yaml \
#     --dataset_dir datasets/ &

# wait

# python ./cs285/scripts/run_hw5_offline.py \
#     -cfg experiments/offline/cql_alpha_experiments/pointmass_medium_cql_10.yaml \
#     --dataset_dir datasets/ &

# New experiments
python ./cs285/scripts/run_hw5_offline.py \
    -cfg experiments/offline/cql_alpha_experiments/pointmass_medium_cql_02.yaml \
    --dataset_dir datasets/ &

python ./cs285/scripts/run_hw5_offline.py \
    -cfg experiments/offline/cql_alpha_experiments/pointmass_medium_cql_04.yaml \
    --dataset_dir datasets/ &

wait

python ./cs285/scripts/run_hw5_offline.py \
    -cfg experiments/offline/cql_alpha_experiments/pointmass_medium_cql_06.yaml \
    --dataset_dir datasets/ &

python ./cs285/scripts/run_hw5_offline.py \
    -cfg experiments/offline/cql_alpha_experiments/pointmass_medium_cql_08.yaml \
    --dataset_dir datasets/ &

wait

python ./cs285/scripts/run_hw5_offline.py \
    -cfg experiments/offline/cql_alpha_experiments/pointmass_medium_cql_1.yaml \
    --dataset_dir datasets/

