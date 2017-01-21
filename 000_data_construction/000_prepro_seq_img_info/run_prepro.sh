# download MSCOCO images
sh get_mscoco_imgs.sh

# get annotation files
annoFile=../../data/coco/annotations/captions_trainval2014.json
if [ -e $annoFile ]
then
  echo '====> Annotations already exist!!'
else
  sh get_annotations.sh
fi

# preprocessing annotations
python prepro_annotation.py

# preprocessing caption labels
stdbuf -oL python prepro_caption.py --input_json ../../data/coco/coco_trainval_raw.json --num_val 5000 --num_test 5000 --output_json ../../data/coco/cocotalk_trainval_img_info.json --output_h5 ../../data/coco/cocotalk_cap_label.h5 2>&1 | tee log_prepro_caption.log
