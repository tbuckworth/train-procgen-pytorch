source /vol/bitbucket/${USER}/train-procgen-pytorch/venvproc/bin/activate
python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train_ilp.py 2>&1 | tee ilp_training.out
