"""
This script should be run from root directory of this codebase:
https://github.com/tylin/coco-caption
"""

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import sys

input_json = sys.argv[1] + '.json'
annFile = 'annotations/captions_val2014.json'
coco = COCO(annFile)
valids = coco.getImgIds()

print '------------------------ input_json :  ',input_json
checkpoint = json.load(open(input_json, 'r'))
preds = checkpoint['val_predictions']

# filter results to only those in MSCOCO validation set (will be about a third)
preds_filt = [p for p in preds if p['image_id'] in valids]
print 'using %d/%d predictions' % (len(preds_filt), len(preds))
json.dump(preds_filt, open(sys.argv[1]+'_tmp.json', 'w')) # serialize to temporary json file.

resFile = sys.argv[1] + '_tmp.json'
cocoRes = coco.loadRes(resFile)
cocoEval = COCOEvalCap(coco, cocoRes)
cocoEval.params['image_id'] = cocoRes.getImgIds()
cocoEval.evaluate()
print '-----Eval--------------- done'

"""
# save score of each image 
imgScore = []
imgEvals = cocoEval.evalImgs
imgid2idx = {}
for i in range(len(imgEvals)) :
    imgId = imgEvals[i]['image_id']
    imgid2idx[imgId] = i

for i in range(len(imgEvals)) :
  imgId = preds[i]['image_id']
  imgScore.append(imgEvals[imgid2idx[imgId]])
  annId = coco.getAnnIds(imgIds=imgId)
  anns = coco.loadAnns(annId)
  gt_caps = []
  for j in range(len(anns)) :
    gt_caps.append(anns[j]['caption'])
  imgScore[-1]['gtCaps'] = gt_caps

json.dump(imgScore, open(sys.argv[1] + '_imgEval.json', 'w'))
print '-----Eval--------------- dump to :  ', sys.argv[1] + '_imgEval.json'
"""

# create output dictionary
out = {}
for metric, score in cocoEval.eval.items():
    out[metric] = score
# serialize to file, to be read from Lua
json.dump(out, open(sys.argv[1] + '_out.json', 'w'))
print '-----Eval--------------- dump to :  ', sys.argv[1] + '_out.json'
