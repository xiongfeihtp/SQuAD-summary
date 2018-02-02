#!/bin/bash
for num in 0 1 2 4 5; do
    CUDA_VISIBLE_DEVICES=1 nohup python3 -u config.py --mode test --load_path /home/xiongfei/R-Net/basic/$num/save/basic_60000.ckpt --run_id $num --model_name basic >test_log &
done
wait
