#!/bin/bash

cargo run -r -p lz78-experiments --bin generate -- \
    --save-path data/outputs/fashion-mnist--repeat-5000--start-at-root.pkl \
    -e fashion-mnist