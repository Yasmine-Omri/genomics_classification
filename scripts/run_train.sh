#!/bin/bash

# cargo run -r -p lz78-experiments --bin train -- \
#     -s data/outputs/wikitext2--repeat-1--start-at-root.pkl \
#     -e wikitext \
#     --start-at-root

cargo run -r -p lz78-experiments --bin train -- \
    -s data/outputs/fashion-mnist--repeat-5000--start-at-root.pkl \
    -e fashion-mnist \
    --repeat 5000 \
    --start-at-root