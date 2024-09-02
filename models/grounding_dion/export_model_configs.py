#!/usr/bin/python3
##        (C) COPYRIGHT Daniel Wang Limited.
##             ALL RIGHTS RESERVED
##
## File       : export_model_configs.py
## Authors    : lqwang@inspur
## Create Time: 2024-06-26:09:00:09
## Description:
## 
##
import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

model_id = "IDEA-Research/grounding-dino-base"
local_model_path = "/home/lqwang/.cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71/"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(local_model_path)
model = AutoModelForZeroShotObjectDetection.from_pretrained(local_model_path).to(device)

model.model.text_backbone.config.save_pretrained("./bert") #save bert config
model.model.backbone.conv_encoder.model.config.save_pretrained("./swin") # save swin config
