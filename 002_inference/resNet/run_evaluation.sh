#!/bin/bash

# generate simulinks
bash gen_simulinks.sh

##### inference with multiple guiding sentence (pretrained model)  
# testall: MSCOCO test evaluation server which contains 40755 images of not available ground truth caption
#stdbuf -oL th eval_res_att_knn_testall.lua -model model/textGuideAtt/res_textGuideAtt.t7 -output_predictions prediction_result/res_predictions_10NN_testall.json 2>&1 | tee log_res_inference_testall.log

# test5000: split constructed by karpathy, which is used widely in image captioning
stdbuf -oL th eval_res_att_knn_test5000.lua -model model/textGuideAtt/res_textGuideAtt.t7 -output_predictions prediction_result/res_predictions_10NN_test5000.json 2>&1 | tee log_res_inference_test5000.log
