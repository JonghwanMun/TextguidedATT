import os
import time
import random
import argparse
import numpy as  np

import cPickle as pkl
import json 
import sys
sys.path.append('./coco-caption')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from evalSentence import evalSentence

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    # required inputs
    parser.add_argument('-NN_info_path',dest='NN_info_path', required=True, type=str, help='path to NN info containing NN imgs, captions and so on')
    parser.add_argument('-output_most_cap',dest='output_most_cap', required=True, type=str, help='output path of most consensus caption')
    parser.add_argument('-output_k_caps',dest='output_k_caps', required=True, type=str, help='output path of top k consensus captions')
    parser.add_argument('-output_all_caps',dest='output_all_caps', required=True, type=str, help='output path of all consensus captions')

    parser.add_argument('-img_info_path',dest='img_info_path', default='data/coco/coco_trainval_raw.json', type=str, help='path of image information containing file path, image Id ...')
    parser.add_argument('-rerank_mode',dest='rerank_mode', type=str, default='cider', help='cider|bleu')
    parser.add_argument('-sIdx', dest='sIdx', type=int, default=0, help='index of first image for print')
    parser.add_argument('-m', dest='m', type=int, default=125, help='the number of NN captions for concensus similarity, it reduces the effect of outlier captions')
    parser.add_argument('-k', dest='k', type=int, default=10, help='the number of top consensus captions')

    opt = parser.parse_args()
    print 'Options are as follows'
    print opt

    # Load data
    with open(opt.img_info_path, 'r') as f :
        img_info = json.load(f)
    print('Img info file is loaded from %s' % opt.img_info_path)
    with open(opt.NN_info_path, 'r') as f :
        NN_info = json.load(f)
    print('NN info file is loaded from %s' % opt.NN_info_path)
    num_test_imgs = len(NN_info)

    # Reranking
    if opt.rerank_mode == 'cider' : print('Reranking based on CIDEr')
    elif opt.rerank_mode == 'bleu' : print('Reranking based on Bleu-4')
    coco = COCO('./coco-caption/annotations/captions_val2014.json')
    cocoEvalSen = evalSentence(coco, useBleu=True, useCider=True)

    NN_captions       = []
    k_captions        = []
    reranked_captions = []

    if os.path.exists(opt.output_most_cap) :
        with open(opt.output_most_cap, 'r') as f:  NN_captions = json.load(f)
        with open(opt.output_k_caps, 'r') as f:  k_captions = json.load(f)
        with open(opt.output_all_caps, 'r') as f:  reranked_captions = json.load(f)
        print('===>Load precomputed NN files')

    sii = len(NN_captions)
    print 'sii (%d)  | num_test_img (%d)' % (sii, num_test_imgs)

    st = time.time()
    #for ii in range(2) : # for testing this code works well
    for ii in range(sii,num_test_imgs) :  # 5000 items

        ii_imgId = NN_info[ii]['imgId']
        ii_nn_caps = NN_info[ii]['NN_captions']
        num_NN = len(NN_info[ii]['NN_captions'])
        dists = np.zeros( (num_NN, num_NN) )
        consensus_similarity = []

        for i in range(num_NN) :
            tCap = ii_nn_caps[i]
            for j in range(i+1, num_NN) :
                rCap = ii_nn_caps[j]

                if opt.rerank_mode == 'cider' :
                    s = cocoEvalSen.eval_cider([tCap], [ [rCap] ])
                if opt.rerank_mode == 'bleu' :
                    s = cocoEvalSen.eval_bleu([tCap], [ [rCap] ])
                dists[i,j] = s
                dists[j,i] = s
            
            # summation for only top m similarity scores, which reduce the error of outlier
            i_sort_idx = np.argsort( -dists[i,:] )
            consensus_similarity.append( np.sum(dists[i, i_sort_idx[0:opt.m]]) )

        rank_idx = np.argsort(-np.asarray(consensus_similarity)).tolist()
        caps = [ NN_info[ii]['NN_captions'][ri] for ri in rank_idx ]
        NN_imgId = [ NN_info[ii]['NN_imgIds'][int(ri)/5] for ri in rank_idx ]
        NN_imgIdx = [ NN_info[ii]['NN_idx'][int(ri)/5] for ri in rank_idx ]
        capIdx = [ int(ri)%5 for ri in rank_idx ]

        NN_captions.append( {'image_id':ii_imgId, 'NN_captions':caps[0], 'NN_imgId':NN_imgId[0], 'NN_imgIdx':NN_imgIdx[0], 'capIdx':capIdx[0] } )
        k_captions.append( {'image_id':ii_imgId, 'NN_captions':caps[0:opt.k], 'NN_imgId':NN_imgId[0:opt.k], 'NN_imgIdx':NN_imgIdx[0:opt.k], 'capIdx':capIdx[0:opt.k] } )
        reranked_captions.append( {'image_id':ii_imgId, 'NN_captions':caps, 'NN_imgId':NN_imgId, 'NN_imgIdx':NN_imgIdx, 'capIdx':capIdx } )

        if ii % 10 == 0 :
            print('NN caption : %s (for %d)' % (caps[0], img_info[opt.sIdx+ii]['id']))
            #for ci in range(5) : print('GT caption : %s' % (img_info[opt.sIdx+ii]['captions'][ci]))
            remaining_time = (time.time()-st) / (ii-sii+1) * (num_test_imgs-ii) / 60.0
            print('reranking %d/%d done (%.3fs) (%.2fm)' % (ii, num_test_imgs, time.time()-st, remaining_time))
            print('--------------------------------------------------------------------------')

        if ii % 1000 == 0 :
            with open(opt.output_most_cap, 'w') as f:  json.dump(NN_captions,f)
            print 'write most consensus caption to %s' % (opt.output_most_cap)
            with open(opt.output_k_caps, 'w') as f:  json.dump(k_captions,f)
            print 'write top k consensus captions to %s' % (opt.output_k_caps)
            with open(opt.output_all_caps, 'w') as f:  json.dump(reranked_captions,f)
            print 'write all consensus captions to %s' % (opt.output_all_caps)

    # save reranked captions
    # format : { imgID:caption }
    with open(opt.output_most_cap, 'w') as f:  json.dump(NN_captions,f)
    print 'write most consensus caption to %s' % (opt.output_most_cap)
    with open(opt.output_k_caps, 'w') as f:  json.dump(k_captions,f)
    print 'write top k consensus captions to %s' % (opt.output_k_caps)
    with open(opt.output_all_caps, 'w') as f:  json.dump(reranked_captions,f)
    print 'write all consensus captions to %s' % (opt.output_all_caps)

