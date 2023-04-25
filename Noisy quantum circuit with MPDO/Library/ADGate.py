"""
Author: weiguo_ma
Time: 04.25.2023
Contact: weiguo.m@iphy.ac.cn
"""
import torch as tc
import numpy as np
from torch import nn
import Library.tools as tools

class TensorGate(nn.Module):
    def __init__(self, requires_grad: bool = False, device: str or int = 'cpu'):
        super(TensorGate, self).__init__()
        self.name = None
        self.tensor = None
        self.rank = None
        self.dimension = None
        self.single = None

        self.dtype = tc.complex128
        self.device = tools.select_device(device)
        self.requires_grad = requires_grad
        self.variational = self.requires_grad

    def i(self):
        self.name = 'I'
        self.tensor = tc.tensor([[1, 0], [0, 1]], dtype=self.dtype)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        self.variational = False
        return self

    def x(self):
        self.name = 'X'
        self.tensor = tc.tensor([[0, 1], [1, 0]], dtype=self.dtype)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        self.variational = False
        return self

    def y(self):
        self.name = 'Y'
        self.tensor = tc.tensor([[0, -1j], [1j, 0]], dtype=self.dtype)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        self.variational = False
        return self

    def z(self):
        self.name = 'Z'
        self.tensor = tc.tensor([[1, 0], [0, -1]], dtype=self.dtype)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        self.variational = False
        return self

    def s(self):
        self.name = 'S'
        self.tensor = tc.tensor([[1, 0], [0, 1j]], dtype=self.dtype)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        self.variational = False
        return self

    def t(self):
        self.name = 'T'
        self.tensor = tc.tensor([[1, 0], [0, (1 + 1j) / np.sqrt(2)]], dtype=self.dtype)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        self.variational = False
        return self

    def h(self):
        self.name = 'H'
        self.tensor = tc.tensor([[1, 1], [1, -1]], dtype=self.dtype) / np.sqrt(2)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        self.variational = False
        return self

    def cz(self):
        self.name = 'CZ'
        self.tensor = tc.tensor([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, -1]], dtype=self.dtype).reshape((2, 2, 2, 2))
        self.rank = 4
        self.dimension = [[2, 2], [2, 2]]
        self.single = False
        self.variational = False
        return self

    def swap(self):
        self.name = 'SWAP'
        self.tensor = tc.tensor([[1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1]], dtype=self.dtype).reshape((2, 2, 2, 2))
        self.rank = 4
        self.dimension = [[2, 2], [2, 2]]
        self.single = False
        self.variational = False
        return self

    def cnot(self):
        self.name = 'CNOT'
        self.tensor = tc.zeros((2, 2, 2, 2), dtype=self.dtype)  # rank-4 tensor
        # Rank-4 tensor CNOT is constructed by its truth table
        self.tensor[0, 0, 0, 0] = 1  #
        self.tensor[0, 1, 0, 1] = 1
        self.tensor[1, 0, 1, 1] = 1
        self.tensor[1, 1, 1, 0] = 1
        self.rank = 4
        self.dimension = [[2, 2], [2, 2]]
        self.single = False
        self.variational = False
        return self

    # Variational gates

    def rx(self, theta: tc.Tensor = None):
        if theta is None:
            theta = tc.randn(1, dtype=self.dtype)
        if isinstance(theta, float) or isinstance(theta, int):
            theta = tc.tensor(theta, dtype=self.dtype)
        theta = theta.to(device=self.device, dtype=self.dtype)

        self.name = 'RX'
        self.tensor = tc.tensor(
            [[tc.cos(theta / 2), -1j * tc.sin(theta / 2)],
             [-1j * tc.sin(theta / 2), tc.cos(theta / 2)]]
            , dtype=self.dtype
        )
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        self.variational = True

        theta.to(device=self.device)
        return self

    def ry(self, theta: tc.Tensor = None):
        if theta is None:
            theta = tc.randn(1, dtype=self.dtype)
        if isinstance(theta, float) or isinstance(theta, int):
            theta = tc.tensor(theta, dtype=self.dtype)
        theta = theta.to(device=self.device, dtype=self.dtype)

        self.name = 'RY'
        self.tensor = tc.tensor([[tc.cos(theta / 2), -tc.sin(theta / 2)],
                                 [tc.sin(theta / 2), tc.cos(theta / 2)]], dtype=self.dtype)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        self.variational = True
        return self

    def rz(self, theta: tc.Tensor = None):
        if theta is None:
            theta = tc.randn(1, dtype=self.dtype)
        if isinstance(theta, float) or isinstance(theta, int):
            theta = tc.tensor(theta, dtype=self.dtype)
        theta = theta.to(device=self.device, dtype=self.dtype)

        self.name = 'RZ'
        self.tensor = tc.tensor([[tc.exp(-1j * theta / 2), 0],
                                 [0, tc.exp(1j * theta / 2)]], dtype=self.dtype)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        self.variational = True
        return self

    def u1(self, theta: tc.Tensor = None):
        if theta is None:
            theta = tc.randn(1, dtype=self.dtype)
        if isinstance(theta, float) or isinstance(theta, int):
            theta = tc.tensor(theta, dtype=self.dtype)
        theta = theta.to(device=self.device, dtype=self.dtype)

        self.name = 'U1'
        self.tensor = tc.tensor([[1, 0], [0, tc.exp(1j * theta)]], dtype=self.dtype)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        self.variational = True
        return self

    def u2(self, phi: tc.Tensor = None, lam: tc.Tensor = None):
        if phi is None:
            phi = tc.randn(1, dtype=self.dtype)
        if lam is None:
            lam = tc.randn(1, dtype=self.dtype)
        if isinstance(phi, float) or isinstance(phi, int):
            phi = tc.tensor(phi, dtype=self.dtype)
        if isinstance(lam, float) or isinstance(lam, int):
            lam = tc.tensor(lam, dtype=self.dtype)
        phi = phi.to(device=self.device, dtype=self.dtype)
        lam = lam.to(device=self.device, dtype=self.dtype)

        self.name = 'U2'
        self.tensor = tc.tensor([[1, -tc.exp(1j * lam)],
                                 [tc.exp(1j * phi), tc.exp(1j * (phi + lam))]]) / np.sqrt(2)
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        self.variational = True
        return self

    def u3(self, theta: tc.Tensor = None, phi: tc.Tensor = None, lam: tc.Tensor = None):
        if theta is None:
            theta = tc.randn(1, dtype=self.dtype)
        if phi is None:
            phi = tc.randn(1, dtype=self.dtype)
        if lam is None:
            lam = tc.randn(1, dtype=self.dtype)
        if isinstance(theta, float) or isinstance(theta, int):
            theta = tc.tensor(theta, dtype=self.dtype)
        if isinstance(phi, float) or isinstance(phi, int):
            phi = tc.tensor(phi, dtype=self.dtype)
        if isinstance(lam, float) or isinstance(lam, int):
            lam = tc.tensor(lam, dtype=self.dtype)
        theta = theta.to(device=self.device, dtype=self.dtype)
        phi = phi.to(device=self.device, dtype=self.dtype)
        lam = lam.to(device=self.device, dtype=self.dtype)

        self.name = 'U3'
        self.tensor = tc.tensor([[tc.cos(theta / 2), -tc.exp(1j * lam) * tc.sin(theta / 2)],
                                 [tc.exp(1j * phi) * tc.sin(theta / 2),
                                  tc.exp(1j * (phi + lam)) * tc.cos(theta / 2)]])
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        self.variational = True
        return self

    def rzz(self, theta: tc.Tensor = None):
        if theta is None:
            theta = tc.randn(1, dtype=self.dtype)
        if isinstance(theta, float) or isinstance(theta, int):
            theta = tc.tensor(theta, dtype=self.dtype)
        theta = theta.to(device=self.device, dtype=self.dtype)

        self.name = 'RZZ'
        self.tensor = tc.tensor(
            [
                [tc.exp(-1j * theta), 0, 0, 0],
                [0, tc.exp(1j * theta), 0, 0],
                [0, 0, tc.exp(1j * theta), 0],
                [0, 0, 0, tc.exp(-1j * theta)],
            ], dtype=self.dtype
        ).reshape((2, 2, 2, 2))
        self.rank = 4
        self.dimension = [[2, 2], [2, 2]]
        self.single = False
        self.variational = True
        return self

    def arbGateSingle(self, tensor: tc.Tensor = None):
        """ Arbitrary Single qubit gate """
        if tensor is None:
            tensor = tc.randn((2, 2), dtype=self.dtype)
        if isinstance(tensor, np.ndarray):
            self.tensor = tc.tensor(tensor, dtype=self.dtype)
        elif isinstance(tensor, tc.Tensor):
            self.tensor = tensor.to(device=self.device, dtype=self.dtype)
        else:
            raise TypeError('Tensor must be a numpy array or a torch tensor')
        self.name = 'ArbGateSingle'
        self.rank = 2
        self.dimension = [2, 2]
        self.single = True
        self.variational = True
        return self

    def arbGateDouble(self, tensor: tc.Tensor = None):
        """ Arbitrary Double qubit gate """
        if tensor is None:
            tensor = tc.randn((2, 2, 2, 2), dtype=self.dtype)
        if isinstance(tensor, np.ndarray):
            self.tensor = tc.tensor(tensor, dtype=self.dtype)
        elif isinstance(tensor, tc.Tensor):
            self.tensor = tensor.to(device=self.device, dtype=self.dtype)
        else:
            raise TypeError('Tensor must be a numpy array or a torch tensor')

        self.name = 'ArbGateDouble'
        self.rank = 4
        self.dimension = [[2, 2], [2, 2]]
        self.single = False
        self.variational = True
        return self