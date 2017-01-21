#!/bin/bash

######################################################################
##### 0. Data Construction
######################################################################
cd 000_data_construction/000_prepro_seq_img_info
bash run_prepro.sh
cd ../001_resNet101_conv_feat
bash run_extract_res.sh
cd ../002_skipthought
bash run_extract_skip.sh
cd ../003_consensus_captions
bash run_obtain_consensus_captions.sh

######################################################################
##### 1. Training Model
######################################################################
cd ../../001_train/resNet
bash run_train_caption.sh

# After train the model, you should move the model to 'model/TextguideAtt' folder with following line:
# mv trained_model/model_name.t7 ../../model/TextguideAtt/res_TextguideAtt.t7

# if you do not want to train model from scratch
# download the pretrained model using following two lines
# cd ../../
# bash get_pretrained_model.sh

######################################################################
##### 2. Testing Model
######################################################################
cd ../../002_inference
bash run_inference.sh
