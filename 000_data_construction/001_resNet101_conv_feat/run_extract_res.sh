#!/bin/bash

# generate simulinks for easy path configuration
bash gen_simulinks.sh

# download the residual 101-layers network
resFile=model/resNet/resnet-101.t7
if [ -f "$resFile" ]  
then
  echo '====> residual network model exist!!'
else
  bash get_resNet101_model.sh
fi


# extract resnet feature
stdbuf -oL th extract_resnet_feat.lua 2>&1 | tee log_extract_res_feat.log
