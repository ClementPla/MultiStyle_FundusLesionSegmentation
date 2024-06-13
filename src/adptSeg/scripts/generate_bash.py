import numpy as np


positions = np.arange(1, 5)
feature_types = ['encoder', 'decoder']
with open('bash_script.sh', 'w') as f:
    print('#!/bin/bash', file=f)
    for position in positions:
        for feature_type in feature_types:
            print(f'python src/adptSeg/scripts/linear_probing.py --position {position} --feature_type {feature_type}', file=f)
    
    
    