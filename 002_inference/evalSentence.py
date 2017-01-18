# This code is written based on eval.py from https://github.com/tylin/coco-caption
# Used to compute similarity between two sentences with cider or bleu

from tokenizer.ptbtokenizer import PTBTokenizer
from cider import Cider
from cider_scorer import CiderScorer
from bleu import Bleu
from bleu_scorer import BleuScorer

import numpy as np

class evalSentence :
    def __init__(self, coco, useBleu=False, useCider=False) :
        self.coco = coco
        self.useBleu = useBleu
        self.useCider = useCider
        self.params = {'image_id': coco.getImgIds()}

        imgIds = self.params['image_id']
        gts = {}
        for imgId in imgIds :
            gts[imgId] = self.coco.imgToAnns[imgId]
        
        if self.useBleu :
            self.b_scorer = BleuScorer()
        if self.useCider :
            self.c_scorer = CiderScorer()

        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)

        for imgId in imgIds :
            ref = gts[imgId]

            assert(type(ref) is list)
            assert(len(ref) > 0)

            if self.useCider :
                self.c_scorer += (None, ref)

        if self.useCider :
            self.c_scorer.compute_doc_freq()
            assert(len(self.c_scorer.ctest) >= max(self.c_scorer.document_frequency.values()))

    def eval_cider(self, test, ref) :
        assert(self.useCider)

        c_score = self.c_scorer.compute_cider(test, ref)
        return np.array(c_score)

    def eval_bleu(self, test, ref) :
        assert(self.useBleu)

        self.b_scorer.reset_list()
        for ts, rs in zip(test, ref) : self.b_scorer += (ts, rs)
        b_score, b_scores = self.b_scorer.compute_score()
        return b_scores[3]  # return bleu_4
