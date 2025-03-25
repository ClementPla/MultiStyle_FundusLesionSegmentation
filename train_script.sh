#!/bin/bash

python -W ignore src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset IDRID
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset FGADR
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset MESSIDOR
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset DDR
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset RETLES
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset IDRID FGADR
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset IDRID MESSIDOR
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset IDRID DDR
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset IDRID RETLES
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset FGADR MESSIDOR
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset FGADR DDR
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset FGADR RETLES
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset MESSIDOR DDR
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset MESSIDOR RETLES
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset DDR RETLES
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset IDRID FGADR MESSIDOR
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset IDRID FGADR DDR
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset IDRID FGADR RETLES
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset IDRID MESSIDOR DDR
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset IDRID MESSIDOR RETLES
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset IDRID DDR RETLES
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset FGADR MESSIDOR DDR
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset FGADR MESSIDOR RETLES
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset FGADR DDR RETLES
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset MESSIDOR DDR RETLES
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset IDRID FGADR MESSIDOR DDR
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset IDRID FGADR MESSIDOR RETLES
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset IDRID FGADR DDR RETLES
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset IDRID MESSIDOR DDR RETLES
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset FGADR MESSIDOR DDR RETLES
python src/fundseg/scripts/train.py --model unet_seresnext50_32x4d --dataset IDRID FGADR MESSIDOR DDR RETLES
