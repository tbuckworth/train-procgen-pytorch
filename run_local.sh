source /vol/bitbucket/${USER}/train-procgen-pytorch/venvproc/bin/activate
python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name coinrun --env_name coinrun --num_levels 100000 --distribution_mode hard --param_name hard-500-impalavq --num_timesteps 200000000 --num_checkpoints 5 --seed 6033 --random_percent 0 --use_wandb --real_procgen
