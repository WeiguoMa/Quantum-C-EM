"""
Author: weiguo_ma
Time: 04.29.2023
Contact: weiguo.m@iphy.ac.cn
"""
from typing import Optional, Union

from torch import complex64, Tensor
from torch import nn

from Library.ADGate import TensorGate
from Library.ADNGate import NoisyTensorGate
from Library.tools import select_device


class AbstractGate(nn.Module):
    def __init__(self, requires_grad: bool = False, ideal: Optional[bool] = None,
                 _lastTrunc: bool = False, device: Union[str, int] = 'cpu', dtype=complex64):
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
        self.variational = None

    def i(self):
        self.variational = False
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).i()
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def x(self):
        self.variational = False
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).x()
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def y(self):
        self.variational = False
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).y()
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def z(self):
        self.variational = False
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).z()
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def h(self):
        self.variational = False
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).h()
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def s(self):
        self.variational = False
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).s()
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def t(self):
        self.variational = False
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).t()
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def cnot(self):
        self.variational = False
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).cnot()
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def swap(self):
        self.variational = False
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).swap()
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def ii(self):
        self.variational = False
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).ii()
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def cx(self):
        self.variational = False
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).cx()
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def cy(self):
        self.variational = False
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).cy()
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def cz(self):
        self.variational = False
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).cz()
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    # Variational gate

    def cp(self, theta: Optional[Tensor] = None):
        self.variational = True
        self.para = theta
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).cp(self.para)
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def rx(self, theta: Optional[Tensor] = None):
        self.variational = True
        self.para = theta
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).rx(self.para)
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def ry(self, theta: Optional[Tensor] = None):
        self.variational = True
        self.para = theta
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).ry(self.para)
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def rz(self, theta: Optional[Tensor] = None):
        self.variational = True
        self.para = theta
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).rz(self.para)
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def u1(self, theta: Optional[Tensor] = None):
        self.variational = True
        self.para = theta
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).u1(self.para)
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def u2(self, lam: Tensor, phi: Tensor):
        self.variational = True
        self.para = [lam, phi]
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).u2(self.para[0], self.para[1])
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def u3(self, theta, phi, lam):
        self.variational = True
        self.para = [theta, phi, lam]
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).u3(self.para[0], self.para[1], self.para[2])
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def rzz(self, theta):
        self.variational = True
        self.para = theta
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).rzz(self.para)
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def arbGateSingle(self, tensor: Tensor):
        self.variational = True
        self.para = tensor
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).arbGateSingle(self.para)
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    def arbGateDouble(self, tensor: Tensor):
        self.variational = True
        self.para = tensor
        self.gate = TensorGate(ideal=self.ideal, requires_grad=self.requires_grad).arbGateDouble(self.para)
        self.name = self.gate.name
        self.single = self.gate.single
        return self

    # Experimental noisy gates

    def czEXP(self, EXPTensor: Tensor):
        self.variational = False
        self.gate = NoisyTensorGate().czEXP(tensor=EXPTensor)
        self.name = self.gate.name
        self.single = self.gate.single
        return self
