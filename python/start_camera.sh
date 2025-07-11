#!/usr/bin/env bash
python3 try2.py \
  --model /usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk \
  --threshold 0.55 \
  --iou 0.5 \
  --max-detections 10 \
