

python train.py --data_aug_ops heavy --optimizer sgd --log_dice False --dice_smooth 0.4 --lr 0.003 --model unet_timm-efficientnet-b4 --dataset all
python train.py --data_aug_ops heavy --optimizer sgd --log_dice False --dice_smooth 0.4 --lr 0.003 --model fpn_timm-efficientnet-b4 --dataset all
