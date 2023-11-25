#!/bin/bash


cd "$(dirname "$0")"
cd ..

python3 model_test.py --dataset KTH --use-sigmoid --img-channels 3 --img-height 120 --img-width 120 --kernel-size 5 --model convlstm --batch-size 8