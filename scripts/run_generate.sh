#!/bin/bash

cargo run -r -p lz78-experiments --bin generate -- \
    --save-path spa_outputs/wikitext \
    -e wikitext