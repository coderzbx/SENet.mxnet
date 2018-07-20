#!/bin/bash

python val_predict.py \
  --dir "/data/deeplearning/lane_detect/kd.senet.assist/data/highway-assist/test" \
  --gpu -1 \
  --model "/data/deeplearning/lane_detect/kd.senet.assist/models/mobilenetv2-main_road-0090.params" \
  --dest_dir "/data/deeplearning/lane_detect/kd.senet.assist/data/highway-assist/test.mobilenetv2.cpu"

