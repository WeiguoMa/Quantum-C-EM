"""
Author: weiguo_ma
Time: 04.29.2023
Contact: weiguo.m@iphy.ac.cn
"""
import torch as tc
from torch import nn
from typing import Optional

from Library.ADGate import TensorGate
from Library.ADNGate import NoisyTensorGate
from Library.tools import select_device


class AbstractGate(nn.Module):
	def __init__(self, requires_grad: bool = True, ideal: Optional[bool] = None,
	             device: str or int = 'cpu', dtype=tc.complex128):
		super(AbstractGate, self).__init__()
		self.requires_grad = requires_grad
		self.device = select_device(device)
		self.dtype = dtype
		self.name = None
		self.single = None

		self.ideal = ideal

		self.para = None
		self.gate = None
		self.variational = self.requires_grad

	def i(self):
		self.variational = False
		self.gate = TensorGate().i()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def x(self):
		self.variational = False
		self.gate = TensorGate().x()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def y(self):
		self.variational = False
		self.gate = TensorGate().y()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def z(self):
		self.variational = False
		self.gate = TensorGate().z()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def h(self):
		self.variational = False
		self.gate = TensorGate().h()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def s(self):
		self.variational = False
		self.gate = TensorGate().s()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def t(self):
		self.variational = False
		self.gate = TensorGate().t()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def cnot(self):
		self.variational = False
		self.gate = TensorGate().cnot()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def swap(self):
		self.variational = False
		self.gate = TensorGate().swap()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def ii(self):
		self.variational = False
		self.gate = TensorGate().ii()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def cx(self):
		self.variational = False
		self.gate = TensorGate().cx()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def cy(self):
		self.variational = False
		self.gate = TensorGate().cy()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def cz(self):
		self.variational = False
		self.gate = TensorGate().cz()
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	# Variational gate

	def rx(self, theta: tc.Tensor = None):
		self.variational = True
		self.para = theta
		self.gate = TensorGate().rx(self.para)
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def ry(self, theta: tc.Tensor = None):
		self.variational = True
		self.para = theta
		self.gate = TensorGate().ry(self.para)
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def rz(self, theta: tc.Tensor = None):
		self.variational = True
		self.para = theta
		self.gate = TensorGate().rz(self.para)
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def u1(self, theta: tc.Tensor = None):
		self.variational = True
		self.para = theta
		self.gate = TensorGate().u1(self.para)
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def u2(self, lam, phi):
		self.variational = True
		self.para = [lam, phi]
		self.gate = TensorGate().u2(self.para[0], self.para[1])
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def u3(self, theta, phi, lam):
		self.variational = True
		self.para = [theta, phi, lam]
		self.gate = TensorGate().u3(self.para[0], self.para[1], self.para[2])
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def rzz(self, theta):
		self.variational = True
		self.para = theta
		self.gate = TensorGate().rzz(self.para)
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def arbGateSingle(self, tensor: tc.Tensor):
		self.variational = True
		self.para = tensor
		self.gate = TensorGate().arbGateSingle(self.para)
		self.name = self.gate.name
		self.single = self.gate.single
		return self

	def arbGateDouble(self, tensor: tc.Tensor):
		self.variational = True
		self.para = tensor
		self.gate = TensorGate().arbGateDouble(self.para)
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