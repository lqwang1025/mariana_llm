#!/usr/bin/python3
##        (C) COPYRIGHT Daniel Wang Limited.
##             ALL RIGHTS RESERVED
##
## File       : compare.py
## Authors    : lqwang@pandora
## Create Time: 2024-10-07:21:08:57
## Description:
## 
##

import numpy as np
import sys
import torch.nn.functional as F
import torch
import torch.nn as nn

def compare(a, b):
    n_a = np.fromfile(a, dtype=np.float32)
    n_b = np.fromfile(b, dtype=np.float32)
    t_n_a = torch.from_numpy(n_a)
    t_n_b = torch.from_numpy(n_b)
    cos_sim = F.cosine_similarity(t_n_a, t_n_b, dim=0)
    loss = nn.MSELoss()
    output = loss(t_n_a, t_n_b)
    print("cos distance:",cos_sim.item(), "MSE:",output.item())

if __name__=="__main__":
    if len(sys.argv) != 3:
        print("usage: python compare a.bin b.bin")
    else:
        compare(sys.argv[1], sys.argv[2])
