"""
Author: weiguo_ma
Time: 04.26.2023
Contact: weiguo.m@iphy.ac.cn
"""
import torch as tc
import numpy as np
from torch import nn
import Library.tools as tools

class TensorQubits(nn.Module):
	def __init__(self):
		super(TensorQubits, self).__init__()
		