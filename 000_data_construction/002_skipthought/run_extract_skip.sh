#!/bin/bash

# generate simulinks for easy path configuration
bash gen_simulinks.sh

# download the weight of skipthought vector model
skipthoughtFile=model/skipthought/uni_gru_params.t7
if [ -f "$skipthoughtFile" ]  
then
  echo '====> skipthought model exist!!'
else
  bash get_skipthought_model.sh
fi

# extract skipthought feature
stdbuf -oL th extract_cap_feat.lua 2>&1 | tee log_extract_cap_feat.log
