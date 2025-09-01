#!/bin/sh

while true
do
    python3 ./check_gpu.py 60
    tools/dist_train.sh configs/mine/parallel/dino-parallel-0612_r50_8xb2_1x_rgbt_tiny.py 1 \
    --work-dir results/test/ --resume
    sleep 60  # 1h=3600
done

