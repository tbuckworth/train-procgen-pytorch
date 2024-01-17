source /vol/bitbucket/${USER}/train-procgen-pytorch/venvproc/bin/activate

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/distill.py --distribution_mode hard --param_name hard-500-$1 --num_checkpoints 10 --seed 6033 --use_wandb --device gpu --wandb_tags learning_rate
