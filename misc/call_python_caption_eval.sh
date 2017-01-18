#!/bin/bash

cd coco-caption
python myeval.py $1
imgEval=$1'_imgEval.json'
cd ../
