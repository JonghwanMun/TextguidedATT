import os
import json

#########################################################################
####  Load annotations
#########################################################################

trainval_img_info_path = '../../data/coco/annotations/captions_trainval2014.json'
test_img_info_path = '../../data/coco/annotations/image_info_test2014.json'
trainval_img_info = json.load( open(trainval_img_info_path, 'r') )
test_img_info = json.load( open(test_img_info_path, 'r') )

#########################################################################
####  preprocessing test annotation
#########################################################################
print 'Preprocessing for test annotations'
out_test = []
loc = 'test2014'
test_imgs = test_img_info['images']
for i,img in enumerate(test_imgs) :
    jimg = {}
    jimg['file_path'] = os.path.join(loc, img['file_name'])
    jimg['id'] = img['id']
    jimg['split'] = 'test'

    out_test.append(jimg)   

# save the annotation for test data
json.dump(out_test, open('../../data/coco/coco_test_raw.json', 'w'))


#########################################################################
####  preprocessing trainval annotation
#########################################################################

# combine all images and annotations together
imgs =  trainval_img_info['images'] #+ train['images']
annots = trainval_img_info['annotations'] #+ train['annotations']

file_name_lists = []
img_id_lists = []
for i in range(len(annots)):
    img_id_lists.append(annots[i]['image_id'])
for i in range(len(imgs)):
    file_name_lists.append(imgs[i]['file_name'])

# for efficiency lets group annotations by image
itoa = {}
for a in annots:
    imgid = a['image_id']
    if not imgid in itoa: itoa[imgid] = []
    itoa[imgid].append(a)

# load the split files
print 'Loading for split of annotations'
train_split   = open('../../data/coco/coco_split/coco_train.txt','r').read().splitlines()
val_split     = open('../../data/coco/coco_split/coco_val.txt','r').read().splitlines()
test_split    = open('../../data/coco/coco_split/coco_test.txt','r').read().splitlines()
restval_split = open('../../data/coco/coco_split/coco_restval.txt','r').read().splitlines()

# sort the annotation in order of splits
print 'Preprocessing for trainval annotations'
out = []
loc = 'val2014'
for i, filename in enumerate(val_split):
    fIdx = file_name_lists.index(filename)
    imgId = imgs[fIdx]['id']

    jimg = {}
    jimg['file_path'] = os.path.join(loc, filename)
    jimg['id'] = imgId
    jimg['fIdx'] = i + 1

    sents = []
    annotsi = itoa[imgId]
    for a in annotsi:
        sents.append(a['caption'])
    jimg['captions'] = sents
    out.append(jimg)
print len(out)
n_item = len(out)

for i, filename in enumerate(test_split):
    fIdx = file_name_lists.index(filename)
    imgId = imgs[fIdx]['id']

    jimg = {}
    jimg['file_path'] = os.path.join(loc, filename)
    jimg['id'] = imgId
    jimg['fIdx'] = i + 1 + n_item

    sents = []
    annotsi = itoa[imgId]
    for a in annotsi:
        sents.append(a['caption'])
    jimg['captions'] = sents
    out.append(jimg)
print len(out)
n_item = len(out)

loc = 'train2014'
for i, filename in enumerate(train_split):
    fIdx = file_name_lists.index(filename)
    imgId = imgs[fIdx]['id']
    jimg = {}
    jimg['file_path'] = os.path.join(loc, filename)
    jimg['id'] = imgId
    jimg['fIdx'] = i + 1 + n_item

    sents = []
    annotsi = itoa[imgId]
    for a in annotsi:
        sents.append(a['caption'])
    jimg['captions'] = sents
    out.append(jimg)
print len(out)
n_item = len(out)

loc = 'val2014'
for i, filename in enumerate(restval_split):
    fIdx = file_name_lists.index(filename)
    imgId = imgs[fIdx]['id']
    jimg = {}
    jimg['file_path'] = os.path.join(loc, filename)
    jimg['id'] = imgId
    jimg['fIdx'] = i + 1 + n_item

    sents = []
    annotsi = itoa[imgId]
    for a in annotsi:
        sents.append(a['caption'])
    jimg['captions'] = sents
    out.append(jimg)
print len(out)

print 'example of preprocessed annotation'
print out[0]

# save the annotation for train & validation data
json.dump(out, open('../../data/coco/coco_trainval_raw.json', 'w'))
