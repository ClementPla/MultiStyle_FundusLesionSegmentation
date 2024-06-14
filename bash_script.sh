#!/bin/bash

python src/adptSeg/scripts/linear_probing.py --position 4 --feature_type decoder

python src/adptSeg/scripts/linear_probing.py --position 0 --feature_type encoder
python src/adptSeg/scripts/linear_probing.py --position 0 --feature_type decoder

python src/adptSeg/scripts/linear_probing.py --position 1 --feature_type encoder
python src/adptSeg/scripts/linear_probing.py --position 1 --feature_type decoder
python src/adptSeg/scripts/linear_probing.py --position 2 --feature_type encoder
python src/adptSeg/scripts/linear_probing.py --position 2 --feature_type decoder
python src/adptSeg/scripts/linear_probing.py --position 3 --feature_type encoder
python src/adptSeg/scripts/linear_probing.py --position 3 --feature_type decoder
python src/adptSeg/scripts/linear_probing.py --position 4 --feature_type encoder

python src/adptSeg/scripts/linear_probing.py --position 5 --feature_type encoder
