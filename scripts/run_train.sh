#!/bin/bash

cargo run -r -p lz78-experiments --bin train -- \
    -s ./spa_outputs/wikitext \
    -e wikitext \
    --data-dir ./data \
    --repeat 10 \
    --start-at-root