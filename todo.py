# TODO:
#  VQ-MHA:
#   internal optimizer for VQ; <- not necessary (or possible) when using EMA updates. Commit loss still relevant
#   FSQ-MHA;
#   MI alpha scaled from 1 - 0;
#   scale up mutual information coefficient;
#   rename mutual information;
#   re-run impala on new platform;
#    if impala doesn't work properly, change ppo to ppo_new and restore original PPO;
#    ...
#  Distill:
#   try cross-entropy loss;
#   try internal optimizer;
#   ...
#  Box-World!:
#   download and install;
#   recreate architecture;
#   do ILP on it;
#   see if it generalizes;