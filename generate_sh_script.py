arch = ['unet']
encoders = ['seresnext50_32x4d']
datasets = ['IDRID', 'FGADR', 'MESSIDOR', 'DDR', 'RETLES']

from itertools import chain, combinations
def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

all_train_configs = list(all_subsets(datasets))[1:]


models = [f'{a}_{e}' for a in arch for e in encoders]

intro = '''#!/bin/bash
'''

with open('train_script.sh', 'w') as f:
    print(intro, file=f)
    for data in all_train_configs:
        for model in models:
            print(f"python src/fundseg/scripts/train.py --model {model} --dataset {' '.join(data)}", file=f)