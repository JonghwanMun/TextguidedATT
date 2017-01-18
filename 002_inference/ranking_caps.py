import os
import time
import argparse
import numpy as  np

import cPickle as pkl
import h5py
import json

import sys
sys.path.append('./coco-caption')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from evalSentence import evalSentence

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    parser.add_argument('-NN_info_path',dest='NN_info_path', required=True, type=str, help='path to NN info containing NN imgs, captions and so on')
    parser.add_argument('-prediction_path',dest='prediction_path', required=True, type=str, help='path to predictions which are hypothesis captions')
    parser.add_argument('-output_ranked_caps',dest='output_ranked_caps', required=True, type=str, help='')
    parser.add_argument('-output_rank_1_cap',dest='output_rank_1_cap', required=True, type=str, help='')
    parser.add_argument('-rerank_mode',dest='rerank_mode', type=str, default='cider', help='cider|bleu')

    parser.add_argument('-eval_after_rerank',dest='eval_after_rerank', action='store_true', default=True, help='after ranking, do eval for ranked captions?')
    parser.add_argument('-m', dest='m', type=int, default=125, help='the number of NN captions for reranking')
    parser.add_argument('-isTest', dest='isTest', action='store_true', help='ranking is done for Test set?')
    parser.add_argument('-sIdx', dest='sIdx', type=int, default=-1, help='start index')
    parser.add_argument('-eIdx', dest='eIdx', type=int, default=-1, help='end index')

    opt = parser.parse_args()
    print 'Options are as follows'
    print opt

    # Load data
    # NN_info and predictions are same order of imgs
    with open(opt.NN_info_path, 'r') as f :
        NN_info = json.load(f)
    with open(opt.prediction_path, 'r') as f :
        preds = json.load(f)
    num_test_imgs = len(preds)

    # remove same captions
    st = time.time()
    predictions = []
    for ii in range(num_test_imgs) :
        clear_caps = []
        check = {}
        for cc in preds[ii]['caption'] :
            if check.get(cc, None) == None :
                clear_caps.append(cc)
                check[cc] = 'true'
        ith_json = {'image_id':preds[ii]['image_id'], 'caption':clear_caps}
        predictions.append(ith_json)
    print 'clearing same captions done (%.2fs)' % (time.time()-st)

    # Reranking
    if opt.rerank_mode == 'cider' : print('Ranking based on CIDEr')
    elif opt.rerank_mode == 'bleu' : print('Ranking based on Bleu-1')
    #coco = COCO('./coco-caption/annotations/captions_train2014.json')
    coco = COCO('./coco-caption/annotations/captions_val2014.json')
    cocoEvalSen = evalSentence(coco, useBleu=True, useCider=True)

    sIdx = 0;   eIdx = num_test_imgs
    if opt.sIdx != -1 : sIdx = opt.sIdx
    if opt.eIdx != -1 : eIdx = opt.eIdx
    print('sIdx (%d)   |   eIdx (%d)' % (sIdx, eIdx) )

    ranked_captions = []
    eval_captions = []

    if os.path.exists(opt.output_ranked_caps) :
        with open(opt.output_ranked_caps, 'r') as f:  ranked_captions = json.load(f)
        with open(opt.output_rank_1_cap, 'r') as f:  eval_captions = json.load(f)
        sIdx = sIdx + len(ranked_captions)
        print('===>Load precomputed ranked files')

    st = time.time()
    #for ii in range(2) : # for testing this code works well
    for ii in range(sIdx, eIdx) :
        #assert NN_info[ii]['imgId'] == predictions[ii]['image_id']

        imgId = predictions[ii]['image_id']
        scores = []
        for pCap in predictions[ii]['caption'] :
            pair_score = []
            for rCap in NN_info[ii]['NN_captions'] :
                if opt.rerank_mode == 'cider' :  s = cocoEvalSen.eval_cider([pCap], [ [ rCap ] ])
                elif opt.rerank_mode == 'bleu' : s = cocoEvalSen.eval_bleu([pCap], [ [ rCap ] ])
                pair_score.append(s[0])
            pair_score.sort(reverse=True)
            scores.append(sum(pair_score[:opt.m]))

        rank_idx = np.argsort(-np.asarray(scores)).tolist()
        rank_1_cap = predictions[ii]['caption'][rank_idx[0]]

        eval_captions.append( {'image_id':imgId, 'caption':rank_1_cap} )
        ranked_captions.append( {'image_id':imgId, 'caption':[ predictions[ii]['caption'][ri] for ri in rank_idx ]} )

        if ii % 100 == 0 : 
            remaining_time = (time.time()-st) / (ii-sIdx+1) * (eIdx-ii) / 60.0
            print('reranking %d/%d done for imgId(%d) (%.2fs) (%.2fm)' % (ii, eIdx, imgId, time.time()-st, remaining_time))

        if ii % 1000 == 0 : 
            with open(opt.output_ranked_caps, 'w') as f:  json.dump(ranked_captions,f)
            print 'write ranked captions to %s' % (opt.output_ranked_caps)
            with open(opt.output_rank_1_cap, 'w') as f:  json.dump(eval_captions,f)
            print 'write rank 1 caption to %s' % (opt.output_rank_1_cap)

    # save ranked captions 
    # format : { imgID:caption }
    with open(opt.output_ranked_caps, 'w') as f:  json.dump(ranked_captions,f)
    print 'write ranked captions to %s' % (opt.output_ranked_caps)
    with open(opt.output_rank_1_cap, 'w') as f:  json.dump(eval_captions,f)
    print 'write rank 1 caption to %s' % (opt.output_rank_1_cap)

    # evaluation ranked captions
    if opt.eval_after_rerank and opt.isTest == False :
        coco = COCO('coco-caption/annotations/captions_val2014.json')
        cocoRes = coco.loadRes(opt.output_rank_1_cap)
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        cocoEval.evaluate()
