#!/bin/bash

cargo run -r -p lz78-experiments --bin generate -- \
    --save-path spa_outputs/c4-realnews \
    --dataset c4 \
    --topk 5\
    -t 0.1 \
    -n 1000 \
    --seed-data "This"

# cargo run -r -p lz78-experiments --bin generate -- \
#     --save-path spa_outputs/shakespeare\
#     --dataset shakespeare \
#     --seed-data "This" \
#     --topk 5\
#     -t 0.1 \
#     -n 800