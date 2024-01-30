arch = ['manet', 'unet']
encoders = ['resnet50', 'se_resnet50', 'se_resnet101', 'mit_b1', 'mit_b2']

models = [f'{a}_{e}' for a in arch for e in encoders]

intro = '''#!/bin/bash 
#
#SBATCH --job-name=RetinalLesionsSegmentation
#SBATCH --output=log.txt
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --partition=liv4d

source /etc/profile.d/modules.sh
module load anaconda3

source activate ~/.conda/envs/torch18
'''

with open('train_script.sh', 'w') as f:
    print(intro, file=f)
    
    
    for model in models:
        print("python train.py --model", model, file=f)