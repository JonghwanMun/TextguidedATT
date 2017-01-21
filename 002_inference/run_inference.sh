#~/bin/bash

# generating prediction result
cd resNet/
bash run_evaluation.sh
cd ../

##### rank the generated cations and select caption of highest score as output
bash gen_simulink.sh

# testall: MSCOCO test evaluation server which contains 40755 images of not available ground truth caption
#stdbuf -oL python ranking_caps.py -NN_info_path data/coco/all_consensus_cap_testall_cider.json -prediction_path resNet/prediction_result/res_predictions_10NN_testall.json -output_ranked_caps output/res_ranked_caps_testall.json -output_rank_1_cap output/res_rank_1_cap_testall.json -isTest 2>&1 | tee log_run_inference_testall.log

# test5000: split constructed by karpathy, which is used widely in image captioning
stdbuf -oL python ranking_caps.py -NN_info_path data/coco/all_consensus_cap_test5000_cider.json -prediction_path resNet/prediction_result/res_predictions_10NN_test5000.json -output_ranked_caps output/res_ranked_caps_test5000.json -output_rank_1_cap output/res_rank_1_cap_test5000.json 2>&1 | tee log_run_inference_test5000.log
