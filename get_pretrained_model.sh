#!/bin/bash

# download the pretrained model with residual network as image feature encoder
wget cvlab.postech.ac.kr/~jonghwan/textGuidedATT/models/res_textGuidedAtt.t7
mv res_textGuidedAtt.t7 ./model/textGuidedAtt/

