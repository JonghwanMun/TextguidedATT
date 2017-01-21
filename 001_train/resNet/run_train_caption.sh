#!/bin/bash

# generate simulinks
bash gen_simulinks.sh

# train the model
stdbuf -oL th train_textGuidedATT.lua -every_vis 30 -learning_rate_decay_start 10 -reg_alpha_lambda 0.002 -reg_alpha_type entropy -use_NN 1 -NN_prob_start_point 1.0 -NN_prob_end_point 1.0 -batch_size 16 -ft_continue -1 -id res_val -gpuid 0 2>&1 | tee log_train.log
