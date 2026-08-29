from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#-- Blocks (self contained for isolation) ----
class CausalSelfAttention(nn.Module):
    