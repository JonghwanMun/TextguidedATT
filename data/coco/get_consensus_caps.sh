#!/bin/bash

echo '====> Download precomputed consensus captions'

wget cvlab.postech.ac.kr/research/captioning/coco/10NN_cap_testall_cider.json
wget cvlab.postech.ac.kr/research/captioning/coco/10NN_cap_valtrainall_cider.json
wget cvlab.postech.ac.kr/research/captioning/coco/all_consensus_cap_testall_cider.json
wget cvlab.postech.ac.kr/research/captioning/coco/all_consensus_cap_test5000_cider.json

mv 10NN_cap_testall_cider.json ../../data/coco/
mv 10NN_cap_valtrainall_cider.json ../../data/coco/
mv all_consensus_cap_testall_cider.json ../../data/coco/
mv all_consensus_cap_test5000_cider.json ../../data/coco/
