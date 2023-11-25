"""
Author: weiguo_ma
Time: 04.30.2023
Contact: weiguo.m@iphy.ac.cn
"""
import torch as tc

from Library.realNoise import czExp_channel
from Library.tools import select_device


class NoisyTensorGate(object):
	def __init__(self, chi: int = None, device: str or int = 'cpu', dtype=tc.complex64):
		self.name = None
		self.tensor = None
		self.rank = None
		self.dimension = None
		self.single = None
		self.ideal = False

		self.device = select_device(device)
		self.dtype = dtype

	def czEXP(self, tensor: tc.Tensor = None):
		self.name = 'CZEXP'
		if tensor is None:  # NO input may cause high memory cost and time cose
			self.tensor = czExp_channel()
		else:
			self.tensor = tensor.to(device=self.device, dtype=self.dtype)
		self.rank = 5
		self.dimension = [[2, 2], [2, 2], ['int[According2EXP]']]
		self.single = False
		return self