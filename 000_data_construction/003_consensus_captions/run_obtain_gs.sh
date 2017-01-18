#!/bin/bash

precomputed=false
if [ "$precomputed" = "true" ]
then
  ##### Download precomputed consensus captions 
  echo '====> Download precomputed consensus captions'

  wget cvlab.postech.ac.kr/research/captioning/coco/10NN_cap_testall_cider.json
  wget cvlab.postech.ac.kr/research/captioning/coco/10NN_cap_valtrainall_cider.json
  wget cvlab.postech.ac.kr/research/captioning/coco/all_consensus_cap_testall_cider.json
  wget cvlab.postech.ac.kr/research/captioning/coco/all_consensus_cap_test5000_cider.json

  mv 10NN_cap_testall_cider.json ../../data/coco/
  mv 10NN_cap_valtrainall_cider.json ../../data/coco/
  mv all_consensus_cap_testall_cider.json ../../data/coco/
  mv all_consensus_cap_test5000_cider.json ../../data/coco/

else
  ##### Construct consensus captions from scratch
  echo '====> Construct consensus captions from scratch'

  ### Generate simulinks for easy path configuration
  bash gen_simulinks.sh

  ### Download feature for computing distance between images
  ### I use the mRNN image feature obtained from https://github.com/mjhucla/mRNN-CR
  ### Actually, any feature can be used. But, the order of imgs should be same with those in img info file.
  trainvalfeat=./feat/mRNN_trainval_feat.h5
  testfeat=./feat/mRNN_test_feat.h5
  if [ -e $trainvalfeat ]
  then
    echo '====> features exist'
  else
    wget cvlab.postech.ac.kr/research/captioning/data/mRNN_trainval_feat.h5
    wget cvlab.postech.ac.kr/research/captioning/data/mRNN_test_feat.h5
    mv mRNN_trainval_feat.h5 feat/
    mv mRNN_test_feat.h5 feat/
  fi

  ### find similar images (kNN) and compute consensus captions for trainval
  stdbuf -oL th find_similar_trainval_img.lua -feat_h5_file ${trainvalfeat} -output_json_path tmp_data/trainval_sim_imgs 2>&1 | tee log_find_similar_img_trainval.log

  n=1
  sIdx=0
  for ii in {1..13}
  do
    stdbuf -oL python find_consensus_caption.py -NN_info_path tmp_data/trainval_sim_imgs_${n}.json -output_most_cap tmp_data/NN_cap_valtrain${n}_cider.json -output_k_caps tmp_data/10NN_cap_valtrain${n}_cider.json -output_all_caps tmp_data/all_cap_valtrain${n}_cider.json -rerank_mode cider -k 10 -sIdx ${sIdx} 2>&1 | tee log_find_consensus_caps_trainval_${n}.log
    n=$((n+1))
    sIdx=$((sIdx+10000))
  done
  python merge_result.py -data_type valtrain
  mv data/coco/all_consensus_cap_valtrainall_cider.json data/coco/all_consensus_cap_test5000_cider.json

  ### find similar images (kNN) and compute consensus captions for test
  stdbuf -oL th find_similar_test_img.lua -feat_h5_file ${trainfeat} -test_feat_h5_file ${testfeat} -output_json_path tmp_data/test_sim_imgs 2>&1 | tee log_find_similar_img_test.log

  n=1
  sIdx=0
  for ii in {1..5}
  do
    stdbuf -oL python find_consensus_caption.py -NN_info_path tmp_data/test_sim_imgs_${n}.json -output_most_cap tmp_data/NN_cap_test${n}_cider.json -output_k_caps tmp_data/10NN_cap_test${n}_cider.json -output_all_caps tmp_data/all_cap_test${n}_cider.json -rerank_mode cider -k 10 -sIdx ${sIdx} 2>&1 | tee log_find_consensus_caps_test_${n}.log
    n=$((n+1))
    sIdx=$((sIdx+10000))
  done
  python merge_result.py -data_type test
fi
