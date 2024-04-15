clear
source /vol/bitbucket/${USER}/train-procgen-pytorch/venvcartpole/bin/activate
python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/create_sh_files.py --n_gpu $1 --execute --hparam_type $2
deactivate
ls -tl scripts/tmp* | head -n$1
