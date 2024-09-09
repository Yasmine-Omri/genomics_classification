#!/bin/bash

# cargo run -r -p lz78-experiments --bin train -- \
#     -s data/outputs/wikitext2--repeat-1--start-at-root.pkl \
#     -e wikitext \
#     --start-at-root

cargo run -r -p lz78-experiments --bin train -- \
    -s spa_outputs/realnewslike \
    -e c4 \
    --data-dir ~/data \
    --start-at-root