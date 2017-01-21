## get mscoco dataset (images)

# http://mscoco.org/dataset/#download
cd ../../data/MSCOCO
# train set images
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
# validation set images
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
# test set images
wget http://msvocds.blob.core.windows.net/coco2014/test2014.zip
# unzip
unzip train2014.zip
unzip val2014.zip
unzip test2014.zip
# rm compressed files
rm -rf train2014.zip
rm -rf val2014.zip
rm -rf test2015.zip
cd ../../000_data_construction/000_prepro_seq_img_info


