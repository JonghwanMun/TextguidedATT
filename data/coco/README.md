# data of captioning for MS COCO dataset

This directory contains two directories:
  - annotations: directory for MSCOCO 2014 annotations
  - coco_split: directory for split constructed by kapathy

This folder contains files:
  - coco_*_raw.json: image and caption information folder for * data (*:trainval|test)
  - cocotalk_trainval_img_info.json: preprocessed image data containing file path, image id ...
  - cocotalk_cap_label.h5: preprocessed caption data containing index information, caption length ...
  - 10NN_cap_*_cider.json: top 10 consensus captions for images, which will be used for inference
  - all_consensus_cap_ *_cider.json: all consensus captions for images, which will be used for ranking the generated captions with multiple guiding sentence
