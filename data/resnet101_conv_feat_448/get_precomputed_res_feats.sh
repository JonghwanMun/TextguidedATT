## get precomputed image features from 101-layer Residual Network

# trainset images
wget http://cvlab.postech.ac.kr/~jonghwan/research/textGuidedATT/feats/resnet_101_conv_448.tar.gz
tar -zxvf resnet_101_conv_448.tar.gz
mv vqa_resnet_101_convfeat_448/* ./
rm -r vqa_resnet_101_convfeat_448
