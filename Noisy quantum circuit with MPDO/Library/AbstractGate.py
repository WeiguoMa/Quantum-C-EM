"""
Author: weiguo_ma
Time: 04.29.2023
Contact: weiguo.m@iphy.ac.cn
"""
from typing import Optional

import torch as tc
from torch import nn

from Library.ADGate import TensorGate
from Library.ADNGate import NoisyTensorGate
from Library.tools import select_device


class AbstractGate(nn.Module):
	def __init__(self, requires_grad: bool = True, ideal: Optional[bool] = None,
	             _lastTrunc: bool = False, device: str or int = 0, dtype=tc.complex128):
		super(AbstractGate, self).__init__()
		self.requires_grad = requires_grad
		self.device = select_device(device)
		self.dtype = dtype
		self.name = None
		self.single = None
		self._lastTruncation = _lastTrunc

		self.ideal = ideal

		self.para = None
		self.gate = None
		self.variational = self.requires_grad

	def i(self):
		self.variational = False
		self.gate = TensorGate(ideal=self.ideal).i()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def x(self):
		self.variational = False
		self.gate = TensorGate(ideal=self.ideal).x()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def y(self):
		self.variational = False
		self.gate = TensorGate(ideal=self.ideal).y()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def z(self):
		self.variational = False
		self.gate = TensorGate(ideal=self.ideal).z()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def h(self):
		self.variational = False
		self.gate = TensorGate(ideal=self.ideal).h()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def s(self):
		self.variational = False
		self.gate = TensorGate(ideal=self.ideal).s()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def t(self):
		self.variational = False
		self.gate = TensorGate(ideal=self.ideal).t()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def cnot(self):
		self.variational = False
		self.gate = TensorGate(ideal=self.ideal).cnot()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def swap(self):
		self.variational = False
		self.gate = TensorGate(ideal=self.ideal).swap()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def ii(self):
		self.variational = False
		self.gate = TensorGate(ideal=self.ideal).ii()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def cx(self):
		self.variational = False
		self.gate = TensorGate(ideal=self.ideal).cx()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def cy(self):
		self.variational = False
		self.gate = TensorGate(ideal=self.ideal).cy()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def cz(self):
		self.variational = False
		self.gate = TensorGate(ideal=self.ideal).cz()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	# Variational gate

	def rx(self, theta: tc.Tensor = None):
		self.variational = True
		self.para = theta
		self.gate = TensorGate(ideal=self.ideal).rx(self.para)
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def ry(self, theta: tc.Tensor = None):
		self.variational = True
		self.para = theta
		self.gate = TensorGate(ideal=self.ideal).ry(self.para)
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def rz(self, theta: tc.Tensor = None):
		self.variational = True
		self.para = theta
		self.gate = TensorGate(ideal=self.ideal).rz(self.para)
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def u1(self, theta: tc.Tensor = None):
		self.variational = True
		self.para = theta
		self.gate = TensorGate(ideal=self.ideal).u1(self.para)
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def u2(self, lam, phi):
		self.variational = True
		self.para = [lam, phi]
		self.gate = TensorGate(ideal=self.ideal).u2(self.para[0], self.para[1])
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def u3(self, theta, phi, lam):
		self.variational = True
		self.para = [theta, phi, lam]
		self.gate = TensorGate(ideal=self.ideal).u3(self.para[0], self.para[1], self.para[2])
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def rzz(self, theta):
		self.variational = True
		self.para = theta
		self.gate = TensorGate(ideal=self.ideal).rzz(self.para)
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def arbGateSingle(self, tensor: tc.Tensor):
		self.variational = True
		self.para = tensor
		self.gate = TensorGate(ideal=self.ideal).arbGateSingle(self.para)
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def arbGateDouble(self, tensor: tc.Tensor):
		self.variational = True
		self.para = tensor
		self.gate = TensorGate(ideal=self.ideal).arbGateDouble(self.para)
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	# Experimental noisy gates

	def czEXP(self, EXPTensor: tc.Tensor = None):
		self.variational = False
		self.gate = NoisyTensorGate().czEXP(tensor=EXPTensor)
		self.name = self.gate.name
		self.single = self.gate.single
		return self